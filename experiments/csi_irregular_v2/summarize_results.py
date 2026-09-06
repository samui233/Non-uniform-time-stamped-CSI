#!/usr/bin/env python3
"""Read completed artifacts only; write a concise Chinese experiment report.

No inference, training, RT, score-based filtering, or fabricated placeholders.
Missing artifacts remain explicitly unfinished. Smoke/candidate directories are
never searched implicitly.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

MODELS=['cru','latent_ode','neural_flow','tpatchgnn']
ORDER=['hold','linear','kalman','harmonic',*MODELS]
LABEL={'hold':'保持','linear':'因果线性拟合','kalman':'连续时间 Kalman',
       'cru':'CRU','latent_ode':'Latent ODE','neural_flow':'Neural Flow','tpatchgnn':'t-PatchGNN','harmonic':'谐波最小二乘'}
PATTERNS=['uniform','random','bursty','dropout']
PATTERN_LABEL={'uniform':'近均匀','random':'随机','bursty':'突发聚簇','dropout':'连续缺测'}


def read_json(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path):
    path=Path(path)
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def mean_exact(values):
    """Do not silently drop a nonfinite/missing seed from an aggregate."""
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):return None
    return float(np.mean(values))


def db(value):
    return 10*math.log10(max(float(value),1e-30)) if value is not None and math.isfinite(float(value)) else None


def fmt(value,places=2):
    return f'{float(value):.{places}f}' if value is not None and math.isfinite(float(value)) else '未完成/异常'


def table(headers,rows):
    if not rows:return ['未完成：没有可用结果。','']
    escape=lambda v:str(v).replace('|','\\|').replace('\n',' ')
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(escape(v) for v in row)+' |' for row in rows],'']


def collect(condition):
    out={}
    for path in sorted(Path(condition).glob('*/summary.json')):
        obj=read_json(path)
        if obj and 'overall' in obj and int(obj['overall'].get('n',0))>0:
            out[path.parent.name]={'result':obj,'path':str(path.resolve()),
                                  'paired':read_json(path.parent/'paired_vs_hold.json')}
    return out


def aggregate(found,group=None,key=None):
    groups=defaultdict(list)
    for name,item in found.items():
        obj=item['result']
        metric=obj['overall'] if group is None else obj.get('groups',{}).get(group,{}).get(key)
        if metric:groups[name.split('_seed')[0]].append((name,metric,item))
    out={}
    for name,entries in groups.items():
        metrics=[e[1] for e in entries]
        linear=mean_exact([m.get('nmse_linear') for m in metrics])
        amp=mean_exact([m.get('amplitude_nmse') for m in metrics])
        seed_db=[db(m.get('nmse_linear')) for m in metrics]
        weighted=mean_exact([10**(m['energy_weighted_nmse_db']/10) if m.get('energy_weighted_nmse_db') is not None else None for m in metrics])
        out[name]={'nmse_linear':linear,'nmse_db':db(linear),'amplitude_nmse_db':db(amp),
            'shape_correlation':mean_exact([m.get('shape_correlation') for m in metrics]),
            'gain_abs_error_db':mean_exact([m.get('gain_abs_error_db') for m in metrics]),
            'energy_weighted_nmse_db':db(weighted),
            'seed_std_db':float(np.std(seed_db,ddof=1)) if len(seed_db)>1 and all(v is not None for v in seed_db) else None,
            'runs':[e[0] for e in entries],'targets_per_run':[m['n'] for m in metrics],
            'sources':[e[2]['path'] for e in entries]}
    return out


def delta(a,b):
    return a-b if a is not None and b is not None else None


def condition_complete(found,seed=17,all_seeds=None):
    required=['hold','linear','kalman','harmonic']+[f'{m}_seed{s}' for m in MODELS for s in (all_seeds or [seed])]
    return all(k in found for k in required)


def write_report(run_root,data_root,output,seeds=(17,29,43)):
    run_root=Path(run_root);data_root=Path(data_root);output=Path(output)
    results=run_root/'results'
    conditions={p.name:collect(p) for p in results.iterdir() if p.is_dir()} if results.exists() else {}
    main=conditions.get('main',{});offgrid=conditions.get('offgrid',{})
    ag={k:aggregate(v) for k,v in conditions.items()}
    overall=ag.get('main',{})
    progress=read_json(run_root/'checkpoints/progress.json') or {}
    completed=progress.get('completed',[])
    coverage=read_json(data_root/'window_coverage.json')
    compression=read_json(data_root/'compression/metadata.json')
    cfg=read_json(data_root/'dataset_config.json') or {}
    manifest=read_json(data_root/'manifest.json') or []
    done=sum((data_root/'routes'/r['traj_id']/'complete.json').exists() for r in manifest)
    selection=read_json(run_root/'stress_model_selection.json') or {}
    selected=selection.get('selected')
    latency=read_csv(results/'latency/timings_summary.csv')
    stress_done=sum(condition_complete(conditions.get(f'{p}_n{n}',{})) for p in PATTERNS for n in [4,8,16,32])
    timing_modes=['none','uniform_times','permuted_association','jitter_5','jitter_20']
    timestamp_done=sum(bool(conditions.get('timestamps_'+p)) for p in timing_modes)
    uniform_done=sum(bool(conditions.get('uniform_trained_'+p)) for p in PATTERNS)
    statuses={'rt_routes':{'complete':done,'planned':len(manifest)},'training_jobs':{'complete':len(completed),'planned':len(MODELS)*len(seeds)},
        'main_complete':condition_complete(main,all_seeds=seeds),'offgrid_complete':condition_complete(offgrid,all_seeds=seeds),
        'stress_conditions_complete':stress_done,'stress_conditions_planned':16,
        'timestamp_conditions_complete':timestamp_done,'timestamp_conditions_planned':5,
        'uniform_conditions_complete':uniform_done,'uniform_conditions_planned':4,
        'latency_conditions_complete':len(latency),'latency_conditions_planned':48}
    derived={'generated_utc':datetime.now(timezone.utc).isoformat(),'run_root':str(run_root.resolve()),
        'data_root':str(data_root.resolve()),'status':statuses,'conditions':ag,'validation_model_selection':selection,
        'aggregation':'Within seed: per-target linear NMSE mean. Across seeds: linear mean then dB. No missing/nonfinite seed is silently dropped.'}
    lines=['# CSI 非均匀时间预测实验结果与分析','',f'自动汇总时间：{derived["generated_utc"]}。仅读取 `{run_root}` 内实际完成的结果；不包含试跑、旧候选射线数据或尚未完成的实验。','',
           '## 完成情况','']
    lines+=table(['阶段','实际状态'],[
        ['有效版本射线数据',f'{done}/{len(manifest)} 条轨迹完成'],
        ['主训练',f'{len(completed)}/{len(MODELS)*len(seeds)} 个任务已记录完成'],
        ['主测试 / 真实 off-grid 测试',f'{"完整" if statuses["main_complete"] else "未完成"} / {"完整" if statuses["offgrid_complete"] else "未完成"}'],
        ['等预算时间模式/数量',f'{stress_done}/16 个条件完整'],
        ['时间戳消融 / 均匀训练对照',f'{timestamp_done}/5、{uniform_done}/4 个条件已有结果'],
        ['GPU 推理时延',f'{len(latency)}/48 个条件已有测量']])
    if not statuses['main_complete']:
        lines+=['**主测试尚未完整，以下已出现的数值只代表已完成部分，不能作为四种方法三种子的最终排名。**','']
    lines+=['## 数据、评价范围与计分方式','']
    if cfg:
        lines += [f'本次配置计划包含 {len(manifest)} 条轨迹，每条 {cfg.get("duration_s","未记录")} 秒，密集 CSI 为 {cfg.get("csi_hz","未记录")} Hz；历史范围 {cfg.get("history_span_s","未记录")} 秒，未来范围 {cfg.get("future_span_s","未记录")} 秒。实际文件完成情况见上表。','']
    if coverage:
        rows=[]
        for split,m in coverage.get('by_split',{}).items():
            rows.append([split,m.get('candidate_cutoffs','未记录'),m.get('eligible_cutoffs','未记录'),fmt(100*m['coverage_fraction'])+'%',m.get('unresolved_frames','未记录')])
        lines+=table(['划分','候选截止点','合格截止点','覆盖率','未解析帧'],rows)
    else:lines+=['窗口资格及覆盖报告尚未生成。','']
    lines+=['主任务只评价整窗射线标签可用时的 CSI 条件预测。未找到路径不视为物理精确零信道；原始数据保留，离线按路径数与有限正能量确定资格，不按增益大小、跳变或模型误差筛选。这不是可观察未来的在线门控，也不覆盖断链预测。','',
        '主要指标为每个未来目标的完整复数矩阵 NMSE，先在线性域逐目标等权平均，再转 dB；越低越好。跨训练种子也在线性域平均后转 dB。能量加权指标按物理幅度计算，幅度 NMSE不评价相位，形状相关为归一化复内积幅值平方。各指标不能互相替代。','']
    if compression:
        lines += [f'复数 PCA 秩为 {compression.get("rank","未记录")}，输入实数维数为 {compression.get("feature_dim","未记录")}；基由训练集拟合，维数选择使用训练/验证重建审计。是否满足预定重建门槛：{compression.get("thresholds_met","未记录")}。完整矩阵误差包含正交补残差；简单基线直接使用完整历史。','']
    else:lines+=['正式 PCA 拟合及审计尚未完成。','']
    lines+=['## 主测试','']
    rows=[]
    for m in ORDER:
        if m not in overall:continue
        r=overall[m]
        seeds_label=', '.join(n.split('_seed')[-1] for n in r['runs']) if m in MODELS else '确定性'
        rows.append([LABEL[m],seeds_label,fmt(r['nmse_db']),fmt(r['amplitude_nmse_db']),fmt(r['shape_correlation'],3),fmt(r['gain_abs_error_db']),fmt(r['energy_weighted_nmse_db'])])
    lines+=table(['方法','已有种子','复数 NMSE/dB','幅度 NMSE/dB','形状相关','增益 MAE/dB','能量加权 NMSE/dB'],rows)
    learned=[m for m in MODELS if m in overall and overall[m]['nmse_db'] is not None]
    if learned:
        best=min(learned,key=lambda m:overall[m]['nmse_db'])
        lines += [f'已完成学习方法中的最低复数 NMSE 为 {LABEL[best]}：{fmt(overall[best]["nmse_db"])} dB。该描述只用于报告结果，次级实验的模型仍由验证集预先选择。','']
        if overall[best]['nmse_linear'] > 1:
            lines += ['**但这个总体误差仍高于恒零预测的0 dB，不能把相对其它方法误差较低解释为已经可靠预测完整CSI。** 恒零预测对每个非零目标的NMSE严格等于1；这里引用的是指标的数学参照，不是新增训练模型。应结合逐目标分布、逐轨迹结果、幅度和形状保真度解读，尤其不能把输出收缩带来的相对误差降低当作恢复了信道动态。详见 [最终结果解读](FINAL_INTERPRETATION.md)。','']
        for reference in ['hold','kalman','harmonic']:
            if reference in overall and overall[reference]['nmse_db'] is not None:
                difference=delta(overall[best]['nmse_db'],overall[reference]['nmse_db'])
                direction='低' if difference<0 else '高'
                lines += [f'相对{LABEL[reference]}，这一点估计{direction} {fmt(abs(difference))} dB。'+('' if difference<0 else '当前结果不支持该学习方法优于这一基线。'),'']
    if 'harmonic' in overall:
        lines += ['谐波最小二乘是附加的经典相位外推对照，不是Time-IMM神经方法：只从历史完整复数CSI和真实时间拟合共享频率及逐元素复系数，频率搜索参数固定，不使用未来观测或验证/测试分数调参。搜索频带以密集CSI时钟为先验，不是任意稀疏非均匀观测的通用Nyquist保证。','']
        comparisons=[]
        for baseline in ['hold','linear','kalman']:
            change=delta(overall['harmonic']['nmse_db'],overall.get(baseline,{}).get('nmse_db'))
            if change is not None:comparisons.append(f'相对{LABEL[baseline]} {change:+.2f} dB')
        if comparisons:lines += ['谐波对照的完整CSI NMSE点估计：'+'；'.join(comparisons)+'。负数表示误差更低。','']
    if learned:
        gaps=[(m,delta(overall[m]['nmse_db'],overall[m]['amplitude_nmse_db'])) for m in learned]
        gaps=[(m,g) for m,g in gaps if g is not None]
        if gaps:
            m,g=max(gaps,key=lambda p:p[1])
            lines += [f'{LABEL[m]}的复数与幅度指标相差 {fmt(g)} dB。幅度指标不能反映相位及复数结构误差；这个 dB 差值不是独立“相位误差”，也不足以判定公共相位处理或采样混叠是原因。','']
    paired=[]
    for name,item in main.items():
        p=item['paired']
        if p and p.get('delta_nmse_db_ci95'):
            lo,hi=p['delta_nmse_db_ci95']
            paired.append([name,fmt(p['delta_nmse_db']),f'[{fmt(lo)}, {fmt(hi)}]',p.get('trajectory_count','未记录')])
    if paired:
        lines+=['与保持法的配对差值（模型减保持，负数更好）及轨迹聚类 bootstrap 95%区间如下。相邻窗口没有被当作独立轨迹。','']
        lines+=table(['运行','差值/dB','95%区间/dB','轨迹数'],paired)
    lines+=['### 不同未来提前量','']
    horizons=[10,20,50,100,200,500];lookup={}
    for name,item in main.items():
        for key in item['result'].get('groups',{}).get('query_s',{}):lookup[round(float(key)*1000)]=key
    horizon_ag={h:aggregate(main,'query_s',lookup[h]) for h in horizons if h in lookup}
    rows=[[LABEL[m],*[fmt(horizon_ag.get(h,{}).get(m,{}).get('nmse_db')) for h in horizons]] for m in ORDER if m in overall]
    lines+=table(['方法',*[f'{h} ms' for h in horizons]],rows)
    derived['main_horizons']=horizon_ag
    if rows:
        lines+=['这是相同历史窗口对不同目标提前量的预测。是否随提前量单调变差应以完整曲线为准，不预设每一步都必须单调。','']
    lines+=['## 真实离网格查询','']
    og=ag.get('offgrid',{})
    lines+=table(['方法','复数 NMSE/dB','幅度 NMSE/dB','每运行目标数'],[
        [LABEL[m],fmt(og[m]['nmse_db']),fmt(og[m]['amplitude_nmse_db']),'/'.join(map(str,og[m]['targets_per_run']))] for m in ORDER if m in og])
    if og:
        lines+=['这些目标在非网格时刻重新执行射线追踪，不来自 CSI 插值。其随机提前量分布与主测试的六个固定提前量不同，因此不能仅用两组总体均值之差衡量“离网格惩罚”；应参考提前量分段结果。','']
        bands=sorted({k for v in offgrid.values() for k in v['result'].get('groups',{}).get('query_band_ms',{})},key=lambda k:float(k.split('-')[0]))
        og_bands={k:aggregate(offgrid,'query_band_ms',k) for k in bands}
        lines+=table(['方法',*[k+' ms' for k in bands]],[[LABEL[m],*[fmt(og_bands[k].get(m,{}).get('nmse_db')) for k in bands]] for m in ORDER if m in og])
        derived['offgrid_bands']=og_bands
    lines+=['## 非均匀历史的影响','',
        '这里采用预先固定种子17。模式/数量比较共享相同截止点和目标；不能使用主测试中不同随机子集的历史数量分组来替代这一受控实验。','',
        '### 固定16个历史观测，改变时间模式','']
    lines+=table(['方法',*[PATTERN_LABEL[p] for p in PATTERNS]],[
        [LABEL[m],*[fmt(ag.get(f'{p}_n16',{}).get(m,{}).get('nmse_db')) for p in PATTERNS]]
        for m in ORDER if any(m in ag.get(f'{p}_n16',{}) for p in PATTERNS)])
    if selected:
        reference=ag.get('uniform_n16',{}).get(selected,{}).get('nmse_db')
        comparisons=[]
        for pattern in PATTERNS[1:]:
            value=ag.get(pattern+'_n16',{}).get(selected,{}).get('nmse_db')
            change=delta(value,reference)
            if change is not None:comparisons.append(f'{PATTERN_LABEL[pattern]}相对近均匀为 {change:+.2f} dB')
        if comparisons:
            lines += [f'验证所选 {LABEL[selected]} 在固定16点下：'+ '；'.join(comparisons)+'。正数表示误差增加，负数表示降低；这里报告点估计，不将模式效应预设为必然有害。','']
    lines+=['### 固定随机时间模式，改变观测数量','']
    lines+=table(['方法','4点','8点','16点','32点'],[
        [LABEL[m],*[fmt(ag.get(f'random_n{n}',{}).get(m,{}).get('nmse_db')) for n in [4,8,16,32]]]
        for m in ORDER if any(m in ag.get(f'random_n{n}',{}) for n in [4,8,16,32])])
    if selected:
        change=delta(ag.get('random_n32',{}).get(selected,{}).get('nmse_db'),ag.get('random_n4',{}).get(selected,{}).get('nmse_db'))
        if change is not None:
            lines += [f'{LABEL[selected]} 从4点增加到32点，复数NMSE变化为 {change:+.2f} dB。'+('此条件下增加观测降低了点估计误差。' if change<0 else '此条件下增加观测未得到更低的点估计误差。')+'该结论固定了时间跨度与目标，不等于更高采样率在所有场景中都会有相同收益。','']
    if stress_done:
        lines+=['完整16条件的结果保存在各条件目录及 `figures/all_conditions.csv`。固定观测预算控制了数量与跨度，但时间布局仍会改变实际看到的信道状态；有差异不等于所有不规则采样都会产生同方向影响。','']
    lines+=['### 时间标签与训练分布','']
    if selected:
        lines += [f'该部分使用验证集选择的 {LABEL.get(selected,selected)}，不是从测试集选择最有利模型。','']
    else:lines+=['验证集模型选择尚未完成，以下次级实验尚不能确定。','']
    names={'none':'真实时间标签','uniform_times':'内部标签改为均匀','permuted_association':'打乱观测—时间对应','jitter_5':'5 ms时间误差','jitter_20':'20 ms时间误差'}
    ref=ag.get('timestamps_none',{}).get(selected,{})
    rows=[]
    for mode in timing_modes:
        r=ag.get('timestamps_'+mode,{}).get(selected)
        if r:rows.append([names[mode],fmt(r['nmse_db']),fmt(delta(r['nmse_db'],ref.get('nmse_db')))])
    lines+=table(['设置','复数 NMSE/dB','相对同种子真实时间/dB'],rows)
    if rows:
        lines+=['正差值表示误差上升。打乱对应属于强破坏性消融，不等同于普通时间抖动；这些固定种子点估计本身不构成统计显著性证明。','']
        uniform_delta=delta(ag.get('timestamps_uniform_times',{}).get(selected,{}).get('nmse_db'),ref.get('nmse_db'))
        if uniform_delta is not None:
            lines += [f'保留观测但将内部时间标签改为均匀后，NMSE变化为 {uniform_delta:+.2f} dB。'+('这给出了本实现使用真实时间标签的正向点估计证据。' if uniform_delta>0 else '该结果没有显示本实现从真实时间标签获得正向点估计收益，不能据此声称时间戳机制已经有效。')+'它不直接证明物理信道是否依赖时间。','']
    rows=[]
    for pattern in PATTERNS:
        mixed=ag.get(pattern+'_n16',{}).get(selected)
        uniform=ag.get('uniform_trained_'+pattern,{}).get(selected)
        if mixed or uniform:
            rows.append([PATTERN_LABEL[pattern],fmt((mixed or {}).get('nmse_db')),fmt((uniform or {}).get('nmse_db'))])
    lines+=table(['测试模式','混合模式训练/dB','均匀模式训练/dB'],rows)
    if rows:lines+=['两列采用同一架构与种子17，历史数量固定16。它检验训练采样分布的适配性，不是两个不同模型结构的比较。','']
    lines+=['## 推理时延','']
    representative=[r for r in latency if int(r['batch_size'])==1 and int(r['history_count'])==16 and int(r['query_count'])==1]
    lines+=table(['方法','预处理p50/ms','模型p50/ms','重建p50/ms','GPU全流程p50/ms','全流程p95/ms'],[
        [LABEL.get(r['model'],r['model']),*[fmt(float(r[k]),3) for k in ['preprocess_ms_median','model_only_ms_median','reconstruct_ms_median','end_to_end_ms_median','end_to_end_ms_p95']]] for r in representative])
    if representative:
        hw=read_json(results/'latency/hardware_and_protocol.json') or {}
        lines += [f'设备：{hw.get("gpu","未记录")}。上表为单样本、16历史点、1未来查询；其它批量与数量见原始时延CSV。GPU全流程包含历史归一化/PCA、网络和CSI重建，但不包括传感器、磁盘、ROS、网络或CPU到GPU传输。单独分阶段测量增加了同步边界，不能将其简单相加当作独立测得的端到端时延。','']
    solver=read_json(results/'solver_convergence/summary.json')
    if solver:
        metrics=solver.get('metrics',{})
        lines+=['## ODE数值检查','',f'在验证集将ODE积分步数加倍，预测差异相对目标能量的均值为 {fmt(metrics.get("difference_over_target_energy",{}).get("mean_db"))} dB；原步数/加倍步数的完整CSI NMSE为 {fmt(metrics.get("coarse_full_csi_nmse",{}).get("mean_db"))}/{fmt(metrics.get("refined_full_csi_nmse",{}).get("mean_db"))} dB。该检查不使用测试集选取积分配置。','']
    else:lines+=['ODE数值步长检查尚未完成或结果未放入本次实验目录。','']
    lines+=['## 结论的边界与文件来源','',
        '本实验是固定地图与基站下的单模态CSI预测，不能据此证明跨地图泛化、实际飞控动力学一致性或视觉模态增益。模型接口接受连续查询时间，实验证据只覆盖本次历史数量和未来时间范围。未解析路径保持为数据可用性问题，不把它们解释成已经验证的物理断链。','',
        '所有表格来自本次实验的 `summary.json`、配对bootstrap文件及时延CSV；正文未引用试跑分数。逐目标结果仍在各条件的 `per_target.csv/.npz`，自动汇总使用的精确数值与文件路径保存于同名JSON。缺失实验保持“未完成”，不以其它窗口、旧checkpoint或人工平滑值补齐。','']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    return statuses


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',default=str(root/'runs/main_v1'))
    p.add_argument('--data-root',default=str(root/'data'))
    p.add_argument('--output',default=str(root/'docs/RESULTS_AND_ANALYSIS.md'))
    a=p.parse_args();status=write_report(a.run_root,a.data_root,a.output)
    print(json.dumps(status,ensure_ascii=False,indent=2));print(a.output)


if __name__=='__main__':main()
