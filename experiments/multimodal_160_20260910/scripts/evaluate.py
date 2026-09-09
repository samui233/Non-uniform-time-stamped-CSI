"""Paired test windows and masks; errors retained per target, not only averages."""
import argparse,json,sys,time
import numpy as np
import torch
from prepare import ROOT,save,digest
from dataset import Histories,loader,move,CFG
from train import terms
sys.path.insert(0,str(ROOT/'model'))
from network_v2 import ScratchFusion

def cases(extended=False,diagnostic=False):
    out={'default':{}}
    for n in [4,8,32]:out[f'csi_{n}']={'csi_count':n}
    for n in [2,16]:out[f'motion_{n}']={'motion_count':n}
    for n in [2,10]:out[f'rgb_{n}']={'rgb_count':n}
    out['all_sparse']={'csi_count':4,'motion_count':2,'rgb_count':2}
    out['all_dense']={'csi_count':32,'motion_count':16,'rgb_count':10}
    if extended:
        for p in ['uniform','bursty','recent','gap']:out['pattern_'+p]={'pattern':p}
        out['wrong_uniform_clocks']={'wrong_clocks':True}
        out['offgrid_queries']={'queries':[.017,.037,.073,.137,.237,.337,.437,.497]}
    if diagnostic:
        out['mismatched_motion']={'mismatched_motion':True}
        out['mismatched_rgb']={'mismatched_rgb':True}
    return out

@torch.no_grad()
def run(model,ds,batch,wrong_clocks=False):
    arrays={k:[] for k in ['numerator','denominator','valid','los','hold_numerator','linear_numerator','pred_change_energy','predicted_energy','amplitude_numerator','phase_aligned_numerator','hold_amplitude_numerator','hold_phase_aligned_numerator','origin','route','category','switch_window','query']}
    for raw in loader(ds,batch,workers=2):
        b=move(raw)
        if wrong_clocks:
            # SAME observations; only replace their internal time spacing.
            for tkey,mkey in [('times','mask'),('motion_times','motion_mask'),('rgb_times','rgb_mask')]:
                for i in range(len(b[tkey])):
                    n=int(b[mkey][i].sum());b[tkey][i,:n]=torch.linspace(float(b[tkey][i,0]),float(b[tkey][i,n-1]),n,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b)
        num,den,valid=terms(p,b);assert torch.isfinite(num).all()
        idx=b['mask'].sum(1)-1;bi=torch.arange(len(idx),device='cuda');last=b['x'][bi,idx];prev=b['x'][bi,idx-1]
        # Baselines always use true timestamps, including the clock-ablation case.
        dt=(raw['times'][torch.arange(len(idx)),idx.cpu()]-raw['times'][torch.arange(len(idx)),idx.cpu()-1]).cuda().clamp_min(1e-6)
        linear=last[:,None]+b['query'][...,None]*(last-prev)[:,None]/dt[:,None,None]
        def diagnostic(prediction,target):
            pc=torch.view_as_complex(prediction.float().contiguous().reshape(*prediction.shape[:-1],-1,2))
            yc=torch.view_as_complex(target.float().contiguous().reshape(*target.shape[:-1],-1,2))
            amp=(pc.abs()-yc.abs()).square().sum(-1)
            aligned=(pc.abs().square().sum(-1)+yc.abs().square().sum(-1)-2*(pc.conj()*yc).sum(-1).abs()).clamp_min(0)
            return amp,aligned
        amp,aligned=diagnostic(p,b['y']);hamp,haligned=diagnostic(last[:,None],b['y'])
        tensors=dict(numerator=num,denominator=den,valid=valid,los=b['target_los'],
            hold_numerator=(last[:,None]-b['y']).square().sum(-1),linear_numerator=(linear-b['y']).square().sum(-1),query=b['query'],
            pred_change_energy=(p-last[:,None]).square().sum(-1),predicted_energy=p.square().sum(-1),
            amplitude_numerator=amp,phase_aligned_numerator=aligned,hold_amplitude_numerator=hamp,hold_phase_aligned_numerator=haligned)
        for key,val in tensors.items():arrays[key].append(val.cpu().numpy())
        for key,bkey in [('origin','origin'),('route','traj_id'),('category','category'),('switch_window','switch_window')]:arrays[key].append(np.asarray(raw[bkey]))
    return {k:np.concatenate(v) for k,v in arrays.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--modalities',required=True);ap.add_argument('--extended',action='store_true');ap.add_argument('--diagnostic',action='store_true');ap.add_argument('--batch',type=int,default=24);a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    path=ROOT/'runs'/a.modalities/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
    model=ScratchFusion(**ck['config']['model']).cuda();model.load_state_dict(ck['model']);model.eval()
    out=ROOT/'results'/a.modalities;out.mkdir(parents=True,exist_ok=True)
    for name,case in cases(a.extended,a.diagnostic).items():
        dest=out/(name+'.npz')
        if dest.exists():continue
        # Motion/image count does not affect the W model. Still preserve a
        # separate evaluated paired case rather than assuming invariance.
        ds=Histories('test',a.modalities,3917,case);tic=time.monotonic()
        data=run(model,ds,a.batch,case.get('wrong_clocks',False));np.savez_compressed(dest,**data)
        valid=data['valid'];mean=(data['numerator']/np.maximum(data['denominator'],1e-20))[valid].mean()
        print('EVAL',a.modalities,name,'NMSE_dB',10*np.log10(mean),'seconds',time.monotonic()-tic,flush=True)
    save(out/'checkpoint.json',dict(path=str(path),sha256=digest(path),epoch=ck['epoch']+1,validation=ck['validation']))
if __name__=='__main__':main()
