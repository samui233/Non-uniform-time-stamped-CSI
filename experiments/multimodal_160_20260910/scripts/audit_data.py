"""One final linear integrity pass; never edits generated observations."""
import json
import numpy as np
from prepare import ROOT,save,digest
from dataset import Histories,collate,ROWS,CFG
def main():
    stats=[];hashes=[]
    cameras=json.loads((ROOT/'cameras.json').read_text())['cameras']
    for r in ROWS:
        out=ROOT/'data'/r['traj_id'];h=np.load(out/'H.npy',mmap_mode='r');t=np.load(out/'csi_time_s.npy');m=np.load(out/'motion.npz');rt=np.load(out/'rgb_time_s.npy');paths=np.load(out/'paths.npz')
        assert h.shape==(2501,64,16) and np.isfinite(h).all()
        assert np.allclose(t,np.arange(2501)/500) and np.allclose(m['time_s'],np.arange(501)/100) and np.allclose(rt,np.arange(101)/20)
        for v in CFG['views']:assert len(list((out/'rgb'/v).glob('*.png')))==101
        features=np.load(out/'features.npy',mmap_mode='r');assert features.shape==(404,64,1024) and np.isfinite(features).all()
        sha=digest(out/'trajectory.npz');assert sha==r['trajectory_sha256'];hashes.append(sha)
        power=np.mean(abs(h)**2,axis=(1,2));zeros=paths['count']==0
        assert np.all(power[zeros]==0)
        point=m['position_m'][::5];inside=[]
        for cam in cameras:
            forward=np.array(cam['forward']);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            delta=point-np.array(cam['position_m']);depth=delta@forward;limit=depth*np.tan(np.deg2rad(cam['fov_deg']/2))
            inside.append((depth>0)&(abs(delta@right)<=limit)&(abs(delta@down)<=limit))
        covered=np.any(inside,axis=0);ray_los=paths['los'][::25];assert len(covered)==len(ray_los)==101
        stats.append(dict(route=r['traj_id'],category=r['category'],split=r['split'],zero_path_frames=int(zeros.sum()),los_fraction=float(paths['los'].mean()),gain_db_min=float(10*np.log10(max(float(power.min()),1e-35))),gain_db_max=float(10*np.log10(max(float(power.max()),1e-35))),
            camera_center_in_frustum_fraction=float(covered.mean()),camera_center_in_frustum_and_rt_los_fraction=float((covered & ray_los).mean())))
    assert len(set(hashes))==160,'Duplicate trajectory files'
    checks=[]
    for split in ['train','val','test']:
        ds=Histories(split,'WMI',17)
        for i in [0,len(ds)//2,len(ds)-1]:
            s=ds[i];assert s['times'][-1]==0 and s['times'].min()>=-.500001 and np.all((s['query']>0)&(s['query']<=.5))
            assert np.isfinite(s['x']).all() and np.isfinite(s['y']).all()
            assert s['motion_times'].max()<=1e-7 and s['rgb_times'].max()<=1e-7
        checks.append(dict(split=split,routes=len(ds.rows),windows_per_epoch=len(ds)))
    save(ROOT/'data_audit.json',dict(complete=True,routes=stats,split_checks=checks,total_csi_frames=160*2501,total_rgb_images=160*404,total_motion_frames=160*501,zero_path_frames=sum(r['zero_path_frames'] for r in stats)))
    print('AUDIT_OK',checks,flush=True)
if __name__=='__main__':main()
