"""Bounded pretraining tests on completed TRAIN recordings only."""
import sys,json
import numpy as np
import torch
from prepare import ROOT,save
from dataset import Histories,collate,move
sys.path.insert(0,str(ROOT/'model'))
from network_v2 import ScratchFusion
from train import metric
def main():
    torch.set_num_threads(2);torch.manual_seed(173)
    ds=Histories('train','WMI',17);ds.rows=[r for r in ds.rows if all((ROOT/'data'/r['traj_id']/p).exists() for p in ['csi_complete.json','rgb_complete.json','features_complete.json'])][:2]
    assert len(ds.rows)==2
    a,b=ds[0],ds[1]
    padded=move(collate([a,b]));single=move(collate([a]));net=ScratchFusion(2048,'WMI').cuda().eval()
    with torch.no_grad():
        pa=net(single);pb=net(padded)[:1]
        assert torch.allclose(pa,pb,atol=2e-6,rtol=2e-4),'padding affected valid sample'
        oldq=single['query'].clone();single['query'].zero_();p0=net(single)
        last=single['x'][:,-1:].float();assert torch.allclose(p0,last.expand_as(p0),atol=1e-7,rtol=1e-6)
        single['query']=oldq
    net.train();p=net(padded);loss=(p-padded['y']).square().mean();loss.backward()
    grads={name:sum(float(v.grad.abs().sum()) for key,v in net.named_parameters() if key.startswith(name) and v.grad is not None) for name in ['csi','motion','rgb','spatial','fusion','layers','decode']}
    assert all(v>0 and np.isfinite(v) for v in grads.values())
    # Dataset draws auxiliary streams independently of CSI count, and vice versa.
    ds.split='test';ds.repeats=1;ds.epoch=0
    ds.case={};base=ds[0]
    ds.case={'csi_count':4};changed=ds[0]
    assert np.array_equal(base['motion_times'],changed['motion_times']) and np.array_equal(base['rgb_times'],changed['rgb_times'])
    ds.case={'rgb_count':2};changed=ds[0]
    assert np.array_equal(base['x'],changed['x']) and np.array_equal(base['times'],changed['times'])
    ds.case={};again=ds[0];assert np.array_equal(base['y'],again['y'])
    zero_net=ScratchFusion(2048,'W').cuda().eval()
    with torch.no_grad():
        zero_net.correction[-1].bias[0]=.5
        single['x'][:,-1].zero_();recovery=zero_net(single)
        assert recovery.abs().max()>1e-7, 'Latest-zero anchor forces zero recovery'
        single['x'].zero_();empty_recovery=zero_net(single)
        assert empty_recovery.abs().max()>1e-7, 'Empty history forces zero output'
    prediction=torch.ones_like(single['y'],requires_grad=True)
    zb={'y':torch.zeros_like(single['y']),'target_valid':torch.zeros_like(single['target_valid'])}
    zero_loss=metric(prediction,zb);zero_loss.backward()
    assert torch.isfinite(zero_loss) and prediction.grad.abs().sum()>0
    save(ROOT/'training_contract_checks.json',dict(padding_max_abs_error=float((pa-pb).abs().max()),zero_horizon_equals_latest=True,nonzero_branch_gradients=grads,independent_stream_sampling=True,deterministic_test_queries=True,latest_zero_recovery_supported=True,all_zero_history_supported=True,zero_target_has_training_gradient=True,scope='completed training recordings only; no test selection'))
    print('TRAINING_CONTRACT_OK',grads,flush=True)
if __name__=='__main__':main()
