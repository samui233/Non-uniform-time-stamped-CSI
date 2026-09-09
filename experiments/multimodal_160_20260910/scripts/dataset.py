"""Route-disjoint continuous-query histories; no future auxiliary information."""
import json
import numpy as np
import torch
from prepare import ROOT
CFG=json.loads((ROOT/'config.json').read_text());ROWS=json.loads((ROOT/'manifest.json').read_text())
ZERO_HISTORY_SCALE=json.loads((ROOT/'normalization.json').read_text())['zero_history_scale']

def choose(rng,times,count,pattern,anchor=False):
    candidates=np.arange(len(times)-(1 if anchor else 0));n=count-(1 if anchor else 0);assert 0<n<=len(candidates)
    u=(times[candidates]+.5)/.5
    if pattern=='uniform':ids=np.rint(np.linspace(0,len(candidates)-1,n)).astype(int)
    else:
        if pattern=='recent':p=.03+np.exp(5*u)
        elif pattern=='bursty':
            centres=rng.uniform(0,1,2);p=.03+sum(np.exp(-.5*((u-c)/.08)**2) for c in centres)
        elif pattern=='gap':
            c=rng.uniform(.2,.8);p=np.where(abs(u-c)<.13,.03,1.)
        else:p=np.ones(len(candidates))
        ids=np.sort(rng.choice(candidates,n,replace=False,p=p/p.sum()))
    if anchor:ids=np.r_[ids,len(times)-1]
    return ids.astype(int)

class Histories(torch.utils.data.Dataset):
    def __init__(self,split,modalities='W',seed=17,case=None):
        self.rows=[r for r in ROWS if r['split']==split];self.split=split;self.modalities=modalities;self.seed=seed;self.epoch=0;self.case=case or {};self.cache={}
        self.cutoffs=np.arange(250,2251,25);self.repeats=CFG['train_resamples_per_window'] if split=='train' else 1
    def __len__(self):return len(self.rows)*len(self.cutoffs)*self.repeats
    def load(self,r):
        rid=r['traj_id']
        if rid not in self.cache:
            out=ROOT/'data'/rid;h=np.load(out/'H.npy',mmap_mode='r');m=np.load(out/'motion.npz')
            motion=np.concatenate([m['position_bs_relative_m']/100,m['velocity_mps']/20],axis=-1).astype(np.float32)
            self.cache[rid]=dict(h=h,motion=motion,mt=m['time_s'],rt=np.load(out/'rgb_time_s.npy'),
                rgb=np.load(out/'features.npy',mmap_mode='r').reshape(101,4,64,1024) if 'I' in self.modalities else None,
                los=np.load(out/'paths.npz')['los'],paths=np.load(out/'paths.npz')['count'])
        return self.cache[rid]
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch if self.split=='train' else 0,index]))
        stream_rngs=[np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch if self.split=='train' else 0,index,tag])) for tag in [100,101,102,103]]
        base=index//self.repeats;ri=base//len(self.cutoffs);ci=base%len(self.cutoffs);row=self.rows[ri];d=self.load(row)
        end=int(self.cutoffs[ci])
        if self.split=='train':end=int(np.clip(end+rng.integers(-12,13),250,2250))
        origin=end/500
        pattern=self.case.get('pattern',rng.choice(['random','uniform','recent','bursty','gap']) if self.split=='train' else 'random')
        def count(key,limits,default):return int(self.case.get(key,rng.integers(limits[0],limits[1]+1) if self.split=='train' else default))
        nc=count('csi_count',CFG['csi_counts'],16);nm=count('motion_count',CFG['motion_counts'],8);nr=count('rgb_count',CFG['rgb_counts'],6)
        histidx=np.arange(end-250,end+1);ht=(histidx-end)/500
        ids=choose(stream_rngs[0],ht,nc,pattern,anchor=True);ht=ht[ids];h=np.array(d['h'][histidx[ids]],dtype=np.complex64).reshape(nc,-1)
        history_energy=float(np.mean(np.sum(abs(h)**2,axis=1)))
        scale=np.sqrt(history_energy) if history_energy>1e-30 else ZERO_HISTORY_SCALE
        h=h/scale
        if self.split=='train':query=(np.arange(6)+stream_rngs[3].uniform(.0001,1,6))*.5/6
        else:query=np.array(self.case.get('queries',CFG['eval_queries_s']),float)
        qpos=end+query*500;lo=np.floor(qpos).astype(int);hi=np.ceil(qpos).astype(int);w=(qpos-lo)[:,None,None]
        y=((1-w)*d['h'][lo]+w*d['h'][hi]).astype(np.complex64).reshape(len(query),-1)/scale
        def stream(times,n,srng):
            eligible=np.flatnonzero((times>=origin-.5-1e-9)&(times<=origin+1e-9));times=times[eligible]-origin
            return eligible[choose(srng,times,min(n,len(times)),pattern)]
        mi=stream(d['mt'],nm,stream_rngs[1]);ii=stream(d['rt'],nr,stream_rngs[2])
        result=dict(x=np.stack([h.real,h.imag],-1).reshape(nc,-1),y=np.stack([y.real,y.imag],-1).reshape(len(query),-1),
            times=ht.astype(np.float32),query=query.astype(np.float32),motion=d['motion'][mi],motion_times=(d['mt'][mi]-origin).astype(np.float32),
            rgb_times=(d['rt'][ii]-origin).astype(np.float32),rgb_indices=ii.astype(np.int64),
            traj_id=row['traj_id'],category=row['category'],origin=origin,scale=float(scale),
            target_valid=(d['paths'][lo]>0)|(d['paths'][hi]>0),target_los=d['los'][np.rint(qpos).astype(int)].astype(bool),
            switch_window=bool(np.any(d['los'][end-250:end+251]!=d['los'][end])),index=index)
        if 'I' in self.modalities:result['rgb']=np.array(d['rgb'][ii])
        if self.case.get('mismatched_motion') or self.case.get('mismatched_rgb'):
            other=self.load(self.rows[(ri+max(1,len(self.rows)//2))%len(self.rows)])
            if self.case.get('mismatched_motion'):result['motion']=other['motion'][mi]
            if self.case.get('mismatched_rgb') and 'I' in self.modalities:result['rgb']=np.array(other['rgb'][ii])
        assert result['times'][-1]==0 and result['motion_times'].max()<=1e-7 and result['rgb_times'].max()<=1e-7
        return result

def collate(samples):
    out={}
    for key,clock in [('x','times'),('motion','motion_times'),('rgb','rgb_times')]:
        lengths=[len(s[clock]) for s in samples];maxlen=max(lengths);mask=np.arange(maxlen)[None]<np.array(lengths)[:,None]
        out[{'x':'mask','motion':'motion_mask','rgb':'rgb_mask'}[key]]=torch.from_numpy(mask)
        for name in [clock]+([key] if key in samples[0] else [])+(['rgb_indices'] if key=='rgb' else []):
            vals=[]
            for s in samples:
                x=s[name];vals.append(np.pad(x,[(0,maxlen-len(x))]+[(0,0)]*(x.ndim-1)))
            out[name]=torch.from_numpy(np.stack(vals))
    for key in ['y','query','target_valid','target_los']:out[key]=torch.from_numpy(np.stack([s[key] for s in samples]))
    for key in ['traj_id','category','origin','scale','switch_window','index']:out[key]=[s[key] for s in samples]
    return out
def loader(ds,batch=8,shuffle=False,workers=2):
    g=torch.Generator().manual_seed(ds.seed+100003*ds.epoch)
    return torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=g,num_workers=workers,collate_fn=collate,pin_memory=True,persistent_workers=False)
def move(b):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
