import json, hashlib, shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/autodl-tmp/csi-visual-occlusion-v1/trajectory_160_preview_v3')
PROJECT=SOURCE.parent
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (ROOT/'manifest.json').exists():return
    rows=json.loads((SOURCE/'manifest.json').read_text());assert len(rows)==160
    bs=json.loads((SOURCE/'config.json').read_text())['bs_position_m']
    cfg=dict(source=str(SOURCE),seed=20260910,bs_position_m=bs,csi_hz=500,motion_hz=100,rgb_hz=20,
        frequencies_hz=((np.rint(np.linspace(0,127,16)).astype(int)-64)*30000).tolist(),
        csi_shape=[64,16],phase_reference='geometric distance; carrier and subcarrier phases; no smoothing',
        receiver_orientation_rad=[0,0,0],rgb_size=400,rgb_fov_deg=100,rgb_down_tilt_deg=10,
        views=['north','east','south','west'],history_s=.5,future_s=.5,cutoff_stride_s=.05,
        train_cutoff_jitter_ms=24,train_resamples_per_window=1,csi_counts=[4,32],motion_counts=[2,16],rgb_counts=[2,10],
        train_query_count=6,eval_queries_s=[.01,.02,.05,.1,.2,.3,.4,.5],
        split_proportions=[.8,.1,.1],latest_csi_at_origin=True,
        image_encoder='frozen ImageNet ResNet50 layer3, native400, adaptive8x8 spatial patches',
        normalization='per-window mean history CSI energy; targets use same history-only scale; position/100m and velocity/20mps',
        model_base='/root/autodl-tmp/csi-irregular-multimodal/experiments/scratch_fusion_20260908/network_v2.py')
    rng=np.random.default_rng(cfg['seed']);counts={}
    for cat in sorted({r['band']+'_'+r['family'] for r in rows}):
        rr=[r for r in rows if r['band']+'_'+r['family']==cat];order=rng.permutation(len(rr))
        n=len(rr);a=int(n*.8);b=int(n*.9);counts[cat]=[a,b-a,n-b]
        for j,idx in enumerate(order):rr[idx]['split']='train' if j<a else 'val' if j<b else 'test'
    for r in rows:
        rid=r['traj_id'];p=SOURCE/'routes'/rid/'trajectory.npz';z=np.load(p)
        assert np.allclose(z['time_s'],np.arange(5001)/1000)
        out=ROOT/'data'/rid;out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,out/'trajectory.npz');idx=np.arange(0,5001,10)
        np.savez(out/'motion.npz',time_s=z['time_s'][idx],position_m=z['position_m'][idx],
            position_bs_relative_m=z['position_m'][idx]-np.array(bs),velocity_mps=z['velocity_mps'][idx],orientation_xyzw=z['orientation_xyzw'][idx])
        r.update(category=r['band']+'_'+r['family'],trajectory_sha256=digest(p))
    phy=json.loads((PROJECT/'debug/site03_bs32_005/geometry_simplification/global_metal_030/physical_config.json').read_text())
    phy['sys_num_subcarriers']=16
    save(ROOT/'physical_config.json',phy);save(ROOT/'config.json',cfg);save(ROOT/'manifest.json',rows)
    save(ROOT/'split_summary.json',counts)
    print('PREPARED',len(rows),counts,flush=True)
if __name__=='__main__':main()
