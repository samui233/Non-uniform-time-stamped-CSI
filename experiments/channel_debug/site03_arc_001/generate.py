"""One curved trajectory with the unchanged site_03 radio configuration."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1];DATA=PROJECT/'dataset'
sys.path[:0]=[str(PROJECT),'/root/autodl-tmp/csi-irregular-benchmark','/root/autodl-tmp/simart-irregular-csi']
def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def bezier(c,u):
    u=np.asarray(u)[:,None];v=1-u
    return v**3*c[0]+3*v*v*u*c[1]+3*v*u*u*c[2]+u**3*c[3]
def tangent(c,u):
    u=np.asarray(u)[:,None];v=1-u
    return 3*v*v*(c[1]-c[0])+6*v*u*(c[2]-c[1])+3*u*u*(c[3]-c[2])
def plan():
    import mitsuba as mi
    from check_geometry import clearance
    cfg=json.loads((DATA/'csi/site_03_route_00/physical_config.json').read_text());scene=mi.load_file(cfg['scene_path'])
    tri=[]
    for mesh in scene.shapes():
        p=mi.traverse(mesh);v=np.asarray(p['vertex_positions']).reshape(-1,3);f=np.asarray(p['faces']).reshape(-1,3);tri.append(v[f])
    tri=np.concatenate(tri);low=tri.min(1);high=tri.max(1)
    ref=np.load(DATA/'routes/site_03_route_00/trajectory.npz');height=float(ref['position_m'][0,2]);speed=float(np.linalg.norm(ref['velocity_mps'][0]))
    attempts=[];selected=None
    for p1,p2 in [((-38,0),(-40,-7)),((-35,0),(-37,-8)),((-32,1),(-36,-10))]:
        c=np.array([[-53,0,height],[*p1,height],[*p2,height],[-40,-20,height]],float);u=np.linspace(0,1,201);p=bezier(c,u)
        # A conservative C2 interpolation error bound for each chord.
        sag=6*max(np.linalg.norm(c[2]-2*c[1]+c[0]),np.linalg.norm(c[3]-2*c[2]+c[1]))/(8*200**2)
        minimum=2.
        for a,b in zip(p[:-1],p[1:]):
            mask=np.all(high>=np.minimum(a,b)-2,axis=1)&np.all(low<=np.maximum(a,b)+2,axis=1)
            if mask.any():minimum=min(minimum,clearance(a,b,tri[mask].astype(float)))
            if minimum<1+sag:break
        attempts.append(dict(control_points_m=c,clearance_lower_bound_m=minimum-sag))
        if minimum>=1+sag:selected=c;break
    save('collision_check.json',dict(attempts=attempts,accepted=selected is not None,method='200 segment-triangle distance checks plus conservative curve-to-chord deviation bound; 1m clearance'))
    if selected is None:raise RuntimeError('No collision-free arc among the three geometry-only candidates')
    dense_u=np.linspace(0,1,50001);dense=bezier(selected,dense_u);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))];length=float(s[-1]);duration=np.ceil(length/speed*500)/500
    t=np.arange(int(round(duration*1000))+1)/1000;u=np.interp(t/duration*length,s,dense_u);p=bezier(selected,u);direction=tangent(selected,u);v=direction/np.linalg.norm(direction,axis=1,keepdims=True)*(length/duration)
    yaw=np.arctan2(v[:,1],v[:,0]);q=np.zeros((len(t),4));q[:,2]=np.sin(yaw/2);q[:,3]=np.cos(yaw/2)
    np.savez(ROOT/'trajectory.npz',time_s=t,position_m=p,velocity_mps=v,orientation_xyzw=q,acceleration_mps2=np.gradient(v,t,axis=0))
    save('config.json',dict(name='site03_arc_001',reference_route='site_03_route_00',bs_position_m=cfg['bs_list'][0]['position'],control_points_m=selected,
        height_m=height,speed_mps=length/duration,length_m=length,duration_s=duration,trajectory_hz=1000,csi_hz=500,coordinate_frame='SimART metres; east +X, north +Y, up +Z',curve='cubic Bezier with arc-length parameterization',clearance_lower_bound_m=minimum-sag))
    save('physical_config.json',cfg);print((ROOT/'config.json').read_text(),flush=True)
def simulate():
    from central_station_study import SimulationConfig,OfflineSionnaSimulator
    from rt_runtime import apply_rt_fix
    from rt_runtime.snapshots_geometry import snapshots_batch
    from union_chain_candidates import UnionChains
    cfg=json.loads((ROOT/'config.json').read_text());physical=json.loads((ROOT/'physical_config.json').read_text())
    assert physical==json.loads((DATA/'csi/site_03_route_00/physical_config.json').read_text())
    sim=OfflineSionnaSimulator(SimulationConfig(**physical));apply_rt_fix(sim);sim.solver._image_method=UnionChains(sim.solver._image_method)
    z=np.load(ROOT/'trajectory.npz');t=z['time_s'][::2];p=z['position_m'][::2];v=z['velocity_mps'][::2]
    if (ROOT/'H.npy').exists():raise FileExistsError('CSI already exists; refusing overwrite')
    h=np.lib.format.open_memmap(ROOT/'H.partial.npy',mode='w+',dtype=np.complex64,shape=(len(t),64,128));counts=[];los=[];tau=[];geo=[];start=time.monotonic()
    for i in range(0,len(t),32):
        hh,r=snapshots_batch(sim,p[i:i+32],v[i:i+32]);h[i:i+len(hh)]=hh[:,0,0]
        for rr in r:counts.append(int(rr['count'][0]));los.append(bool(rr['los'][0]));tau.append(float(rr['tau_min_s'][0]));geo.append(float(rr['tau_geometry_s'][0]))
        print(f'CSI {min(i+32,len(t))}/{len(t)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    assert np.isfinite(h).all();h.flush();del h;(ROOT/'H.partial.npy').rename(ROOT/'H.npy');np.save(ROOT/'time_s.npy',t);np.savez(ROOT/'paths.npz',count=counts,los=los,tau_min_s=tau,tau_geometry_s=geo)
    save('complete.json',dict(frames=len(t),shape=[len(t),64,128],zero_path_frames=int(sum(np.array(counts)==0)),elapsed_s=time.monotonic()-start,
        reference_config_identical=True,candidate_search='mixed_chain_union',code_sha256=hashlib.sha256((PROJECT/'union_chain_candidates.py').read_bytes()).hexdigest(),phase_reference='geometric distance, carrier and subcarriers',smoothed=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['plan','csi']);args=ap.parse_args();plan() if args.stage=='plan' else simulate()
