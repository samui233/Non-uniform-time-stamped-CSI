"""Four paused-state external cameras; source-grid timestamps, lossless PNG."""
import os,sys,json,time,math,signal,subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
import numpy as np
from prepare import ROOT,save
sys.path.insert(0,'/root/autodl-tmp/simart-irregular-csi/experiments/four_view_capture')
import capture_four_views as c

def main():
    cfg=json.loads((ROOT/'config.json').read_text());rows=json.loads((ROOT/'manifest.json').read_text())
    names=cfg['views'];size=cfg['rgb_size'];fov=cfg['rgb_fov_deg'];tilt=math.radians(cfg['rgb_down_tilt_deg'])
    bs=np.array(cfg['bs_position_m']);outroot=ROOT/'capture_runtime';outroot.mkdir(exist_ok=True)
    settings=json.loads((c.RUNTIME/'simart-home/Documents/AirSim/settings.json').read_text())
    settings.update(ViewMode='NoDisplay',SubWindows=[],ExternalCameras={},Recording={'RecordOnMove':False,'RecordInterval':1.,'Cameras':[]})
    for n in names:
        settings['ExternalCameras'][n]={'X':0.,'Y':0.,'Z':-10.,'Pitch':0.,'Roll':0.,'Yaw':0.,'CaptureSettings':[{'ImageType':0,'Width':size,'Height':size,'FOV_Degrees':fov}]}
    sp=outroot/'settings.json';save(sp,settings)
    proc=c.start(sp,outroot);pool=ThreadPoolExecutor(max_workers=4);pending=deque();start=time.monotonic()
    def write_png(folder,i,pixels):
        folder.mkdir(parents=True,exist_ok=True)
        if not c.cv2.imwrite(str(folder/f'{i:04d}.png'),pixels,[c.cv2.IMWRITE_PNG_COMPRESSION,1]):raise RuntimeError('PNG write failed')
    try:
        deadline=time.monotonic()+240
        while not c.ready():
            if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('UE startup failed; see capture_runtime/ue.log')
            time.sleep(1)
        client=c.airsim.MultirotorClient(timeout_value=120);client.confirmConnection();client.enableApiControl(True,vehicle_name='SimpleFlight');client.simPause(True)
        city=client.simGetObjectPose('BigCityCombined')
        assert np.allclose([city.position.x_val,city.position.y_val,city.position.z_val],[105.3,0,0],atol=.001)
        cams=[]
        for n in names:
            direction=np.array(c.DIRECTIONS[n])*math.cos(tilt)+np.array([0,0,-math.sin(tilt)])
            a=c.pos_to_air(bs);v=c.M[:3,:3].T@direction;yaw=math.atan2(v[1],v[0]);pitch=math.atan2(-v[2],np.linalg.norm(v[:2]))
            pose=c.airsim.Pose(c.airsim.Vector3r(*a),c.airsim.to_quaternion(pitch,0,yaw))
            client.simSetCameraPose(n,pose,external=True);client.simSetCameraFov(n,fov,external=True)
            info=client.simGetCameraInfo(n,external=True);c.verify_pose(info.pose,pose);assert abs(info.fov-fov)<.01
            cams.append(dict(name=n,position_m=bs.tolist(),forward=direction.tolist(),fov_deg=fov,resolution=[size,size]))
        save(ROOT/'cameras.json',dict(cameras=cams,airsim_to_scene=c.M.tolist(),source='static map calibration retained including 1m Y correction'))
        req=[c.airsim.ImageRequest(n,c.airsim.ImageType.Scene,False,False) for n in names]
        for ri,row in enumerate(rows):
            out=ROOT/'data'/row['traj_id']
            if (out/'rgb_complete.json').exists():continue
            z=np.load(out/'trajectory.npz');tt=np.arange(101)/20;indices=np.arange(0,5001,50)
            assert np.allclose(z['time_s'][indices],tt)
            stamps=[];errors=[];tic=time.monotonic()
            for fi,idx in enumerate(indices):
                pose=c.pose_to_air(z['position_m'][idx],z['orientation_xyzw'][idx]);assert client.simSetObjectPose('SimpleFlight',pose,True)
                if fi==0:
                    for _ in range(3):client.simGetImages(req,external=True)
                responses=client.simGetImages(req,external=True);assert len(responses)==4
                errors.append(c.verify_pose(client.simGetObjectPose('SimpleFlight'),pose)[0]);stamps.append([int(r.time_stamp) for r in responses])
                for n,r in zip(names,responses):
                    assert r.camera_name==n and (r.width,r.height)==(size,size)
                    pixels=np.frombuffer(r.image_data_uint8,np.uint8).reshape(size,size,3).copy()
                    assert pixels.std()>1
                    pending.append(pool.submit(write_png,out/'rgb'/n,fi,pixels))
                    if len(pending)>=16:pending.popleft().result()
                if fi%20==0:save(ROOT/'rgb_progress.json',dict(route=row['traj_id'],route_index=ri,frame=fi,total_routes=len(rows),elapsed_s=time.monotonic()-start))
            while pending:pending.popleft().result()
            np.save(out/'rgb_time_s.npy',tt);np.save(out/'rgb_engine_timestamps_ns.npy',np.array(stamps,dtype=np.int64))
            save(out/'rgb_complete.json',dict(frames=101,images=404,elapsed_s=time.monotonic()-tic,max_pose_error_m=max(errors),source_trajectory_sha256=row['trajectory_sha256'],synchronization='four views share one paused state and a common trajectory timestamp, not engine wall time'))
            print('RGB',ri+1,len(rows),row['traj_id'],f'{time.monotonic()-tic:.1f}s',flush=True)
        save(ROOT/'rgb_complete.json',dict(routes=len(rows),elapsed_s=time.monotonic()-start))
    finally:
        pool.shutdown(wait=True)
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
if __name__=='__main__':main()
