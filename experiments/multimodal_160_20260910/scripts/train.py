"""Four independent random-initialized predictors, validation-only selection."""
import argparse,json,time,math,sys,hashlib
import numpy as np
import torch
from prepare import ROOT,save,digest
from dataset import Histories,loader,move,CFG
sys.path.insert(0,str(ROOT/'model'))
from network_v2 import ScratchFusion

def atomic_torch(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def terms(p,b):
    y=b['y'].float();num=(p.float()-y).square().sum(-1);den=y.square().sum(-1)
    valid=b['target_valid'] & (den>1e-20)
    return num,den,valid
def metric(p,b):
    num,den,valid=terms(p,b)
    # Zero targets have no NMSE denominator. Retain their error in the
    # history-normalized coordinate instead of dropping those targets.
    return torch.where(valid,num/den.clamp_min(1e-20),num).mean()
@torch.no_grad()
def evaluate(model,ds,batch,workers=2):
    model.eval();values=[];zero_errors=[]
    for raw in loader(ds,batch,workers=workers):
        b=move(raw)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b)
        num,den,valid=terms(p,b);values.append((num/den.clamp_min(1e-20))[valid].cpu().numpy())
        zero_errors.append(num[~valid].cpu().numpy())
    x=np.concatenate(values);assert np.isfinite(x).all()
    zero=np.concatenate(zero_errors)
    return dict(nmse=float(x.mean()),nmse_db=float(10*np.log10(x.mean())),median_nmse_db=float(10*np.log10(np.median(x))),targets=len(x),zero_targets=len(zero),zero_target_normalized_error=float(zero.mean()) if len(zero) else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--modalities',choices=['W','WM','WI','WMI'],required=True)
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--accumulation',type=int,default=8)
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--min-epochs',type=int,default=15);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    tr=Histories('train',a.modalities,17);va=Histories('val',a.modalities,2917)
    if a.smoke:
        completed=[r for r in tr.rows if (ROOT/'data'/r['traj_id']/'csi_complete.json').exists() and (ROOT/'data'/r['traj_id']/'features_complete.json').exists()]
        assert completed,'Need at least one completed training recording'
        tr.rows=completed[:1];tr.repeats=1;tr.cutoffs=tr.cutoffs[:a.batch*3];va.rows=tr.rows;va.cutoffs=tr.cutoffs[:8]
    args=dict(features=2048,modalities=a.modalities,d=32,layers=2)
    model=ScratchFusion(**args).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=3e-3)
    out=ROOT/('smoke' if a.smoke else 'runs')/a.modalities;out.mkdir(parents=True,exist_ok=True)
    cfg=dict(args=vars(a),model=args,data=CFG,normalization=json.loads((ROOT/'normalization.json').read_text()),training_loss='Per-target NMSE for nonzero targets, history-normalized squared error for zero targets; mean over all targets',train_windows_per_epoch=len(tr),parameters=sum(p.numel() for p in model.parameters()),
        predictor_initialization='all random, no pretrained CSI checkpoint',initial_sha256=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'dataset.py',ROOT/'train.py',ROOT/'model/network.py',ROOT/'model/network_v2.py']},
        train_routes=[r['traj_id'] for r in tr.rows],validation_routes=[r['traj_id'] for r in va.rows])
    if (out/'complete.json').exists():print('COMPLETE_ALREADY',out);return
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg,'Resume config mismatch'
    else:save(out/'config.json',cfg)
    start,stale,best,best_epoch,elapsed=0,0,float('inf'),-1,0.
    def factor(e):return (e+1)/2 if e<2 else .03+.97*.5*(1+math.cos(math.pi*(e-2)/max(1,a.epochs-2)))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler'])
        start=ck['epoch']+1;best=ck['best'];best_epoch=ck['best_epoch'];stale=ck['stale'];elapsed=ck['elapsed_s']
        torch.set_rng_state(ck['rng_cpu']);torch.cuda.set_rng_state(ck['rng_cuda'])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.cuda()
        if ck['finished']:start=a.epochs
    clock=time.monotonic();print('START',a.modalities,'PARAMS',cfg['parameters'],'WINDOWS',len(tr),flush=True)
    for epoch in range(start,1 if a.smoke else a.epochs):
        tr.epoch=epoch;model.train();total=count=0;epstart=time.monotonic();last_report=epstart
        dl=loader(tr,a.batch,True,a.workers)
        for step,raw in enumerate(dl):
            b=move(raw)
            if step%a.accumulation==0:opt.zero_grad(set_to_none=True);group=0
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b)
            loss=metric(p,b);assert torch.isfinite(loss),'nonfinite loss'
            effective=a.batch*a.accumulation;(loss*len(p)/effective).backward();group+=len(p)
            if (step+1)%a.accumulation==0 or step+1==len(dl):
                if group!=effective:
                    for param in model.parameters():
                        if param.grad is not None:param.grad.mul_(effective/group)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step()
            total+=float(loss.detach())*len(p);count+=len(p)
            if time.monotonic()-last_report>30:
                row=dict(epoch=epoch+1,step=step+1,steps=len(dl),loss=total/count,elapsed_s=elapsed+time.monotonic()-clock,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                save(out/'progress.json',row);print('STEP',a.modalities,json.dumps(row),flush=True);last_report=time.monotonic()
        val=evaluate(model,va,a.batch,a.workers)
        if val['nmse']<best:
            best=val['nmse'];best_epoch=epoch;stale=0
            atomic_torch(out/'best.pt',dict(model=model.state_dict(),config=cfg,epoch=epoch,validation=val))
        else:stale+=1
        sch.step();finished=a.smoke or epoch+1>=a.epochs or (epoch+1>=a.min_epochs and stale>=a.patience)
        duration=elapsed+time.monotonic()-clock
        row=dict(epoch=epoch+1,train_loss=total/count,validation=val,best_val_nmse_db=float(10*np.log10(best)),best_epoch=best_epoch+1,elapsed_s=duration,epoch_s=time.monotonic()-epstart,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_torch(out/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,elapsed_s=duration,finished=finished,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()))
        save(out/'progress.json',row);print('EPOCH',a.modalities,json.dumps(row),flush=True)
        if finished:break
    save(out/'complete.json',dict(best_epoch=best_epoch+1,best_val_nmse_db=float(10*np.log10(best)),elapsed_s=elapsed+time.monotonic()-clock))
if __name__=='__main__':main()
