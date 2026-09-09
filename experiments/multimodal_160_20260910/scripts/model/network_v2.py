"""One targeted revision: element-shared, phase-equivariant value-path forecasting.

Retains asynchronous history fusion, but removes coefficient-ID memorization.
Sharing over complex coefficients is motivated by the element-wise prediction
discussion in https://eprints.soton.ac.uk/510410/ . This is not that paper's ODE.
The internal phase coordinate is INVERTED before output; data/targets unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from network import Attention, TimeCode


class SharedTemporalAttention(Attention):
    """Compute the identical time-bias MLP once per window, not per coefficient."""
    def forward(self,q,kv,mask,qt,kt,repeats=1):
        N,Q,D=q.shape;B=N//repeats
        dt=(qt.reshape(B,repeats,Q)[:,0,:,None]-kt.reshape(B,repeats,-1)[:,0,None,:]).float()/.5
        bias=self.relative(torch.stack([dt,dt.abs()],-1)).permute(0,3,1,2)
        bias=bias[:,None].expand(B,repeats,self.heads,Q,kv.shape[1]).reshape(N,self.heads,Q,kv.shape[1])
        bias=bias.masked_fill(~mask[:,None,None],float('-inf'))
        def split(x):return x.reshape(N,-1,self.heads,D//self.heads).transpose(1,2)
        k,v=self.kv(self.knorm(kv)).chunk(2,-1)
        out=F.scaled_dot_product_attention(split(self.q(self.qnorm(q))),split(k),split(v),
            attn_mask=bias.to(k.dtype),dropout_p=self.dropout if self.training else 0.)
        h=q+self.out(out.transpose(1,2).reshape(N,Q,D))
        return h+self.ff(h)


class ScratchFusion(nn.Module):
    def __init__(self,features,modalities='WMI',revision=2,d=32,layers=2,groups=None):
        super().__init__()
        self.features,self.modalities,self.revision=features,modalities,revision
        self.scale=math.sqrt(features/2)
        self.time=TimeCode(d)
        self.csi=nn.Sequential(nn.Linear(5,d),nn.GELU(),nn.LayerNorm(d))
        self.global_csi=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.LayerNorm(d))
        self.motion=nn.Sequential(nn.Linear(6,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        # Preserve all 64 patch positions rather than pooling them into two tokens.
        self.rgb=nn.Linear(2048,8)
        self.spatial=nn.Linear(64*8,d)
        self.camera_id=nn.Parameter(torch.randn(4,d)*.02)
        self.motion_attention=Attention(d,dropout=.05)
        self.rgb_attention=Attention(d,dropout=.05)
        self.fusion=nn.Sequential(nn.Linear(3*d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.layers=nn.ModuleList([SharedTemporalAttention(d,dropout=.05) for _ in range(layers)])
        self.anchor=nn.Linear(1,d)
        self.decode=SharedTemporalAttention(d,dropout=.05)
        self.q_weights=nn.Linear(d,16)
        self.k_weights=nn.Linear(d,8)
        self.value_gain=nn.Linear(d,2)
        self.correction=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
        self.multiplier=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2))
        nn.init.zeros_(self.value_gain.weight)
        with torch.no_grad():self.value_gain.bias.copy_(torch.tensor([1.,0.]))
        for h in [self.correction[-1],self.multiplier[-1]]:
            nn.init.normal_(h.weight,std=.001);nn.init.zeros_(h.bias)

    def encode(self,b):
        x=torch.view_as_complex(b['x'].float().contiguous().reshape(*b['x'].shape[:-1],-1,2))
        B,T,K=x.shape;mask=b['mask'];times=b['times']
        last=x[torch.arange(B,device=x.device),mask.sum(1)-1]
        amp=last.abs()
        dominant=last[torch.arange(B,device=x.device),amp.argmax(1)]
        history=x.reshape(B,-1)
        history_peak=history[torch.arange(B,device=x.device),history.abs().argmax(1)]
        dominant=torch.where(dominant.abs()>1e-12,dominant,history_peak)
        fallback=torch.where(dominant.abs()>1e-12,
            dominant.conj()/dominant.abs().clamp_min(1e-12),torch.ones_like(dominant))
        unit=torch.where(amp>1e-7,last.conj()/amp.clamp_min(1e-12),fallback[:,None])
        canonical=x*unit[:,None]
        anchor=last*unit
        delta=canonical-anchor[:,None]
        local=torch.cat([torch.view_as_real(canonical),torch.view_as_real(delta),
                         amp[:,None,:,None].expand(B,T,K,1)],-1)*self.scale
        h=self.csi(local)+self.time(times)[:,:,None]
        inner=(x*last[:,None].conj()).sum(-1)
        stats=torch.stack([inner.real,inner.imag,x.abs().square().sum(-1),delta.abs().square().sum(-1)],-1)
        q=self.global_csi(stats)+self.time(times)
        mc,vc=torch.zeros_like(q),torch.zeros_like(q)
        if 'M' in self.modalities:
            mk=self.motion(b['motion'].float())+self.time(b['motion_times'])
            mc=(self.motion_attention(q,mk,b['motion_mask'],times,b['motion_times'])-q)*b['motion_mask'].any(1)[:,None,None]
        if 'I' in self.modalities:
            raw=b['rgb'].float();diff=torch.zeros_like(raw)
            dt=(b['rgb_times'][:,1:]-b['rgb_times'][:,:-1]).clamp_min(.01)
            diff[:,1:]=(raw[:,1:]-raw[:,:-1])*(.1/dt)[:,:,None,None,None]
            # Only current image spatial statistics, no dataset/route means.
            centered=raw-raw.mean(-2,keepdim=True)
            vk=self.spatial(self.rgb(torch.cat([centered,diff],-1)).flatten(-2))
            B,R,V,D=vk.shape
            rt=b['rgb_times'][:,:,None].expand(B,R,V).reshape(B,-1)
            rm=b['rgb_mask'][:,:,None].expand(B,R,V).reshape(B,-1)
            vk=(vk+self.camera_id).reshape(B,-1,D)+self.time(rt)
            vc=(self.rgb_attention(q,vk,rm,times,rt)-q)*rm.any(1)[:,None,None]
        h=h+self.fusion(torch.cat([q,mc,vc],-1))[:,:,None]
        h=h.transpose(1,2).reshape(B*K,T,-1)
        tt=times[:,None].expand(B,K,T).reshape(B*K,T)
        mm=mask[:,None].expand(B,K,T).reshape(B*K,T)
        for layer in self.layers:h=layer(h,h,mm,tt,tt,repeats=K)
        return h,last,unit,delta,tt,mm

    def forward(self,b):
        h,last,unit,delta,tt,mm=self.encode(b)
        B,K=last.shape;T=h.shape[1];Q=b['query'].shape[1];D=h.shape[-1]
        qt=b['query'][:,None].expand(B,K,Q).reshape(B*K,Q)
        clock=self.time(b['query'])[:,None].expand(B,K,Q,-1).reshape(B*K,Q,-1)
        q=self.anchor(last.abs().reshape(B*K,1,1)*self.scale)+clock
        q=self.decode(q,h,mm,qt,tt,repeats=K)
        qw=self.q_weights(q).reshape(B*K,Q,2,8)
        kw=self.k_weights(h)
        score=torch.einsum('nqhd,ntd->nqht',qw,kw)/math.sqrt(8)
        score=score.masked_fill(~mm[:,None,None],float('-inf'))
        weights=score.softmax(-1)
        signed=weights[:,:,0]-weights[:,:,1]
        values=delta.transpose(1,2).reshape(B*K,T)
        with torch.autocast('cuda',enabled=False):
            real=torch.einsum('nqt,nt->nq',signed.float(),values.real.float())
            imag=torch.einsum('nqt,nt->nq',signed.float(),values.imag.float())
        g=self.value_gain(q).float()
        value=torch.complex(real,imag)*torch.complex(g[...,0],g[...,1])
        free=self.correction(q).float()*(.1/self.scale)
        value=value+torch.complex(free[...,0],free[...,1])
        change=value.reshape(B,K,Q).transpose(1,2)*unit.conj()[:,None]
        a=self.multiplier(q.reshape(B,K,Q,D).mean(1)).float()
        alpha=torch.complex(a[...,0],a[...,1])[...,None]
        # Exact original CSI coordinate; the measured-history value path is part
        # of the complex correction, not an externally supplied forecast.
        p=last[:,None]+(b['query'].float()/.1)[...,None]*(alpha*last[:,None]+change)
        return torch.view_as_real(p).flatten(-2)
