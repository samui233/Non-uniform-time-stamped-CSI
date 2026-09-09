"""Only a fallback for entirely zero histories; derived from TRAIN recordings."""
import json
import numpy as np
from prepare import ROOT,save
def main():
    rows=json.loads((ROOT/'manifest.json').read_text());energy=[]
    for r in rows:
        if r['split']!='train':continue
        h=np.load(ROOT/'data'/r['traj_id']/'H.npy',mmap_mode='r')
        e=np.sum(np.abs(h)**2,axis=(1,2));energy.append(e[e>1e-30])
    e=np.concatenate(energy)
    save(ROOT/'normalization.json',dict(zero_history_scale=float(np.sqrt(np.median(e))),positive_train_frames=len(e),
        policy='Use mean history energy normally. If the entire history has zero energy, use the square root of median nonzero training-frame matrix energy. No validation/test statistics or future-window information.'))
    print('ZERO_HISTORY_SCALE',np.sqrt(np.median(e)),flush=True)
if __name__=='__main__':main()
