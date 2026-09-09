import os,sys,json,time,argparse
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from prepare import ROOT,PROJECT,save
sys.path[:0]=[str(PROJECT),'/root/autodl-tmp/csi-irregular-benchmark','/root/autodl-tmp/simart-irregular-csi']
from central_station_study import SimulationConfig,OfflineSionnaSimulator
from rt_runtime import apply_rt_fix
from rt_runtime.snapshots_geometry import cfr_reference_numpy,geometric_delay,path_delay_minimum,_numpy
from union_complete_candidates import UnionComplete

def snapshots(sim,p,v,frequencies):
    from sionna.rt import Receiver
    for name in list(sim.scene.receivers):sim.scene.remove(name)
    for k,(pos,vel) in enumerate(zip(p,v)):
        sim.scene.add(Receiver(name=f'pose_{k:04}',position=pos.tolist(),orientation=[0,0,0],velocity=vel.tolist()))
    names=['max_depth','samples_per_src','synthetic_array','los','specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region','seed']
    kwargs={k:getattr(sim.cfg,k) for k in names};kwargs['max_num_paths_per_src']=sim.cfg.max_num_paths_per_src*len(p)
    paths=sim.solver(sim.scene,**kwargs)
    geo=geometric_delay(sim,p);mini=path_delay_minimum(paths)
    h=cfr_reference_numpy(paths,frequencies,float(_numpy(sim.scene.frequency).reshape(-1)[0]),geo)[:,0,0].astype(np.complex64)
    valid=_numpy(paths.valid,bool)
    if valid.ndim==5:valid=valid[:,0,:,0,:]
    inter=sim.squeeze_1x1_interactions(_numpy(paths.interactions))
    los=np.any(valid & ~np.any(inter!=0,axis=0),axis=-1)[:,0]
    return h,valid.sum(-1)[:,0],los,mini[:,0],geo[:,0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);a=ap.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text());rows=json.loads((ROOT/'manifest.json').read_text())[a.worker::a.workers]
    if a.limit:rows=rows[:a.limit]
    sim=OfflineSionnaSimulator(SimulationConfig(**json.loads((ROOT/'physical_config.json').read_text())))
    apply_rt_fix(sim);sim.solver._image_method=UnionComplete(sim.solver._image_method)
    start=time.monotonic()
    for ri,r in enumerate(rows):
        out=ROOT/'data'/r['traj_id']
        if (out/'csi_complete.json').exists():continue
        z=np.load(out/'trajectory.npz');t=z['time_s'][::2];p=z['position_m'][::2];v=z['velocity_mps'][::2]
        h=np.lib.format.open_memmap(out/'H.partial.npy',mode='w+',dtype=np.complex64,shape=(len(t),64,16))
        cnt=[];los=[];tau=[];geo=[];tic=time.monotonic()
        for i in range(0,len(t),32):
            hh,cc,ll,tt,gg=snapshots(sim,p[i:i+32],v[i:i+32],cfg['frequencies_hz'])
            assert hh.shape==(min(32,len(t)-i),64,16) and np.isfinite(hh).all()
            h[i:i+len(hh)]=hh;cnt.extend(cc);los.extend(ll);tau.extend(tt);geo.extend(gg)
            if i%320==0:
                save(ROOT/f'csi_progress_{a.worker}.json',dict(route=r['traj_id'],route_index=ri,worker_total=len(rows),frame=i,total_frames=len(t),elapsed_s=time.monotonic()-start))
        h.flush();del h;(out/'H.partial.npy').replace(out/'H.npy');np.save(out/'csi_time_s.npy',t)
        np.savez(out/'paths.npz',count=cnt,los=los,tau_min_s=tau,tau_geometry_s=geo)
        save(out/'csi_complete.json',dict(shape=[len(t),64,16],frequency_hz=500,elapsed_s=time.monotonic()-tic,zero_path_frames=int((np.array(cnt)==0).sum()),trajectory_sha256=r['trajectory_sha256']))
        print('CSI',a.worker,ri+1,len(rows),r['traj_id'],f'{time.monotonic()-tic:.1f}s',flush=True)
    save(ROOT/f'csi_complete_{a.worker}.json',dict(routes=len(rows),elapsed_s=time.monotonic()-start))
if __name__=='__main__':main()
