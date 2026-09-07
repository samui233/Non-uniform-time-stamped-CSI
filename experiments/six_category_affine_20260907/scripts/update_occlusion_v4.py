"""Publish actual v4 results; never substitute v3 values for missing experiments."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/root/autodl-tmp/csi-irregular-benchmark/runs/occlusion_v4_affine_s17')
DATA=Path('/root/autodl-tmp/csi-irregular-benchmark/data_occlusion_v4_nearby')
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows,fields=None):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def replace(name,new):
    p=ROOT/'tables'/name;old=read(p);fields=list(old[0]);kept=[r for r in old if r['family']!='occlusion']
    write(p,kept+[{k:r.get(k,'') for k in fields} for r in new],fields)
def main():
    summaries=[];horizons=[];events=[];sources={}
    for p in sorted((RUN/'results').glob('*/occlusion/*/summary.json')):
        d=json.loads(p.read_text());case=p.parents[2].name;model=p.parent.name
        summaries.append(dict(family='occlusion',case=case,model=model,**d['overall']))
        horizons.extend(dict(family='occlusion',case=case,model=model,query_s=float(q),**v) for q,v in d['groups'].get('query_s',{}).items())
        events.extend(dict(family='occlusion',case=case,model=model,event=q,**v) for q,v in d['groups'].get('event_category',{}).items())
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    replace('overall.csv',summaries);replace('by_horizon.csv',horizons)
    write(ROOT/'tables/occlusion_v4_by_event.csv',events)
    training=[];configs={}
    for p in sorted((RUN/'checkpoints/occlusion').glob('*/complete.json')):
        d=json.loads(p.read_text());c=d['config'];model=c['backbone'];configs[model]=c
        training.append(dict(family='occlusion',model=model,seed=17,best_epoch=d['best_epoch'],val_nmse_db=d['validation']['full']['nmse_db'],elapsed_s=d['elapsed_s'],peak_memory_gb=d['peak_memory_gb'],parameters=c['params']))
    replace('training.csv',training)
    paired=[];bins=[]
    with np.load(RUN/'results/main/occlusion/hold/per_target.npz') as z:b={k:z[k] for k in z.files}
    for model in configs:
        with np.load(RUN/f'results/main/occlusion/{model}/per_target.npz') as a:z={k:a[k] for k in a.files}
        assert all(np.array_equal(z[k],b[k]) for k in ['sample_id','traj_id','query_s'])
        paired.append(dict(family='occlusion',model=model,targets=len(z['nmse']),nmse_db=10*np.log10(z['nmse'].mean()),hold_nmse_db=10*np.log10(b['nmse'].mean()),delta_vs_hold_db=10*np.log10(z['nmse'].mean()/b['nmse'].mean()),fraction_targets_better_than_hold=(z['nmse']<b['nmse']).mean()))
        for label,mask in [('drop_10db',z['target_over_last_db']<=-10),('middle',abs(z['target_over_last_db'])<10),('rise_10db',z['target_over_last_db']>=10)]:
            if mask.any():bins.append(dict(family='occlusion',model=model,bin=label,n=int(mask.sum()),nmse_db=10*np.log10(z['nmse'][mask].mean()),gain_mae_db=z['gain_abs_error_db'][mask].mean(),shape_correlation=z['shape_correlation'][mask].mean(),fraction_of_total_nmse=z['nmse'][mask].sum()/z['nmse'].sum()))
    replace('paired_vs_hold.csv',paired);replace('energy_change_groups.csv',bins)
    replace('before_after.csv',[]);replace('latency.csv',[])
    manifest=json.loads((DATA/'manifest.json').read_text());positions=[];speeds=[];times=[]
    with np.load(ROOT/'data/trajectory_map.npz') as a:d={k:a[k] for k in a.files}
    chunks=[];ts=[];ids=[];families=[];splits=[]
    for i,f in enumerate(d['families']):
        if f=='occlusion':continue
        s=slice(d['offsets'][i],d['offsets'][i+1]);chunks.append(d['xyz'][s]);ts.append(d['time_s'][s]);ids.append(d['traj_ids'][i]);families.append(f);splits.append(d['splits'][i])
    md=json.loads((ROOT/'data/trajectory_map_metadata.json').read_text())
    md['trajectory_sha256']={k:v for k,v in md['trajectory_sha256'].items() if not k.startswith('occlusion')}
    for r in manifest:
        p=DATA/'routes'/r['traj_id']/'trajectory.npz'
        with np.load(p) as a:
            xyz=a['position_m'];t=a['time_s'];positions.append(xyz);speeds.append(np.linalg.norm(a['velocity_mps'],axis=1));chunks.append(xyz[::10]);ts.append(t[::10])
        ids.append(r['traj_id']);families.append('occlusion');splits.append(r['split']);md['trajectory_sha256'][r['traj_id']]=hashlib.sha256(p.read_bytes()).hexdigest()
    np.savez_compressed(ROOT/'data/trajectory_map.npz',xyz=np.concatenate(chunks),time_s=np.concatenate(ts),offsets=np.r_[0,np.cumsum([len(x) for x in chunks])],traj_ids=np.array(ids),families=np.array(families),splits=np.array(splits),**{k:d[k] for k in ['map_polygons','map_heights','bs_position_m']})
    md.update(occlusion_source=str(DATA),map_scope='First five unchanged; occlusion_v4_000..059, assigned spatial splits',occlusion_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest())
    (ROOT/'data/trajectory_map_metadata.json').write_text(json.dumps(md,indent=2)+'\n')
    xyz=np.concatenate(positions);v=np.concatenate(speeds)
    row=dict(family='occlusion',routes=60,duration_s=2.5,trajectory_hz=1000,csi_hz=500,z_min_m=xyz[:,2].min(),z_max_m=xyz[:,2].max(),speed_min_mps=v.min(),speed_max_mps=v.max())
    for split in ['train','val','test']:
        rr=[r for r in manifest if r['split']==split];row[split+'_routes']=len(rr);row[split+'_groups']=len({r['channel_group'] for r in rr})
    replace('dataset_summary.csv',[row])
    meta=json.loads((ROOT/'tables/experiment_metadata.json').read_text())
    meta.update(occlusion_version='v4_nearby',occlusion_source=str(RUN),occlusion_available_cases=sorted({r['case'] for r in summaries}),occlusion_missing_policy='Not evaluated; no v3 substitution',old_new_comparison='First five only: no matched previous-model v4 evaluation',occlusion_training_configs=configs,occlusion_result_sha256=sources)
    (ROOT/'tables/experiment_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    p=ROOT/'tables/latency_metadata.json';lm=json.loads(p.read_text());lm['scope']='First five only; old occlusion timing removed, v4 not measured';p.write_text(json.dumps(lm,indent=2)+'\n')
    print('Updated v4 tables/map; cases:',meta['occlusion_available_cases'])
if __name__=='__main__':main()
