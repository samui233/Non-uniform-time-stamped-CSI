"""Stream frozen native400 image features as recordings finish."""
import sys,time,json,hashlib
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from prepare import ROOT,save
sys.path.insert(0,'/root/autodl-tmp/csi-irregular-multimodal/vendor')
from torchvision.models import resnet50
class Images(torch.utils.data.Dataset):
    def __init__(self,files):self.files=files
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        with Image.open(self.files[i]) as im:a=np.array(im.convert('RGB'))
        return torch.from_numpy(a).permute(2,0,1)
def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    cfg=json.loads((ROOT/'config.json').read_text());rows=json.loads((ROOT/'manifest.json').read_text())
    wp=Path('/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth');sha=hashlib.sha256(wp.read_bytes()).hexdigest();assert sha.startswith('19c8e357')
    net=resnet50(weights=None);net.load_state_dict(torch.load(wp,map_location='cpu',weights_only=False))
    net=torch.nn.Sequential(*list(net.children())[:7]).cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    start=time.monotonic()
    for i,r in enumerate(rows):
        out=ROOT/'data'/r['traj_id']
        if (out/'features_complete.json').exists():continue
        while not (out/'rgb_complete.json').exists():time.sleep(5)
        files=[out/'rgb'/v/f'{j:04d}.png' for j in range(101) for v in cfg['views']]
        data=torch.utils.data.DataLoader(Images(files),batch_size=24,num_workers=2,pin_memory=True)
        arr=np.lib.format.open_memmap(out/'features.partial.npy',mode='w+',dtype=np.float16,shape=(404,64,1024));pos=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for raw in data:
                x=(raw.cuda(non_blocking=True).float()/255-mean)/std
                y=torch.nn.functional.adaptive_avg_pool2d(net(x),8).flatten(2).transpose(1,2)
                arr[pos:pos+len(y)]=y.float().cpu().numpy();pos+=len(y)
        arr.flush();del arr;(out/'features.partial.npy').replace(out/'features.npy')
        save(out/'features_complete.json',dict(shape=[101,4,64,1024],stored_shape=[404,64,1024],weights_sha256=sha,input='native400 RGB; ImageNet normalization; no crop or dataset means',encoder='frozen ResNet50 layer3 adaptive8x8'))
        print('FEATURES',i+1,len(rows),r['traj_id'],f'{time.monotonic()-start:.1f}s',flush=True)
    save(ROOT/'features_complete.json',dict(routes=160,elapsed_s=time.monotonic()-start))
if __name__=='__main__':main()
