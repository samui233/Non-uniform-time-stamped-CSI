"""Restartable local supervisor. Never changes data/model settings after launch."""
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
from prepare import ROOT,save
PY='/root/miniconda3/envs/SimART/bin/python'
ROWS=json.loads((ROOT/'manifest.json').read_text())
def running(script,worker=None):
    for p in psutil.process_iter(['cmdline']):
        cmd=p.info['cmdline'] or []
        if str(ROOT/script) in cmd and (worker is None or ('--worker' in cmd and cmd[cmd.index('--worker')+1]==str(worker))):return True
    return False
def spawn(name,cmd):
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    f=(ROOT/'logs'/f'{name}.log').open('a')
    p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env);f.close();return p
def main():
    start=time.monotonic();attempts={}
    while True:
        counts={k:sum((ROOT/'data'/r['traj_id']/file).exists() for r in ROWS) for k,file in [('csi','csi_complete.json'),('rgb','rgb_complete.json'),('features','features_complete.json')]}
        save(ROOT/'workflow_status.json',dict(stage='data',counts=counts,elapsed_s=time.monotonic()-start))
        if min(counts.values())==160:break
        specs=[]
        for i in range(8):
            if any(not (ROOT/'data'/r['traj_id']/'csi_complete.json').exists() for r in ROWS[i::8]):
                specs.append((f'csi_{i}','generate_csi.py',i,[PY,str(ROOT/'generate_csi.py'),'--worker',str(i),'--workers','8']))
        if counts['rgb']<160:specs.append(('rgb','capture_rgb.py',None,['/usr/bin/python3',str(ROOT/'capture_rgb.py')]))
        if counts['features']<160:specs.append(('features','cache_features.py',None,[PY,str(ROOT/'cache_features.py')]))
        for name,script,worker,cmd in specs:
            if not running(script,worker):
                attempts[name]=attempts.get(name,0)+1
                if attempts[name]>3:raise RuntimeError(f'{name} repeatedly failed; inspect log')
                spawn(name,cmd);print('RESTART',name,flush=True)
        print('DATA',counts,flush=True);time.sleep(20)
    if not (ROOT/'normalization.json').exists():subprocess.run([PY,str(ROOT/'prepare_normalization.py')],check=True)
    subprocess.run([PY,str(ROOT/'audit_data.py')],check=True)
    for pair in [('W','WM'),('WI','WMI')]:
        jobs=[]
        for m in pair:
            if (ROOT/'runs'/m/'complete.json').exists():continue
            cmd=[PY,str(ROOT/'train.py'),'--modalities',m,'--batch','32','--accumulation','2']
            jobs.append((m,spawn('train_'+m,cmd)))
        while any(p.poll() is None for m,p in jobs):
            save(ROOT/'workflow_status.json',dict(stage='training',models={m:p.poll() for m,p in jobs},elapsed_s=time.monotonic()-start));time.sleep(20)
        for m,p in jobs:
            if p.returncode!=0:raise RuntimeError('Training failed: '+m)
    for pair in [('W','WM'),('WI','WMI')]:
        jobs=[(m,spawn('eval_'+m,[PY,str(ROOT/'evaluate.py'),'--modalities',m,'--batch','24'])) for m in pair]
        for m,p in jobs:
            if p.wait()!=0:raise RuntimeError('Evaluation failed: '+m)
    subprocess.run([PY,str(ROOT/'analyze.py')],check=True)
    decision=json.loads((ROOT/'analysis/assessment.json').read_text())
    if decision['ideal_for_extended_tests']:
        for pair in [('W','WM'),('WI','WMI')]:
            jobs=[(m,spawn('eval_'+m,[PY,str(ROOT/'evaluate.py'),'--modalities',m,'--extended','--batch','24'])) for m in pair]
            for m,p in jobs:
                if p.wait()!=0:raise RuntimeError('Extended evaluation failed: '+m)
        subprocess.run([PY,str(ROOT/'analyze.py')],check=True)
    else:
        for pair in [('W','WM'),('WI','WMI')]:
            jobs=[(m,spawn('eval_'+m,[PY,str(ROOT/'evaluate.py'),'--modalities',m,'--diagnostic','--batch','24'])) for m in pair]
            for m,p in jobs:
                if p.wait()!=0:raise RuntimeError('Diagnostic evaluation failed: '+m)
        subprocess.run([PY,str(ROOT/'analyze.py')],check=True)
    save(ROOT/'workflow_status.json',dict(stage='results_ready_for_review',elapsed_s=time.monotonic()-start,assessment=decision))
    print('RESULTS_READY',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'workflow_failure.json',dict(error=repr(e)));raise
