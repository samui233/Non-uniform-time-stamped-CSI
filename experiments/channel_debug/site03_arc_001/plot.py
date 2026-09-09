"""Unsmoothed figures for one arc; all plotted numbers retained alongside plots."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
def main(src,out):
    out.mkdir(parents=True,exist_ok=True);cfg=json.loads((src/'config.json').read_text());z=np.load(src/'trajectory.npz');h=np.load(src/'H.npy',mmap_mode='r');t=np.load(src/'time_s.npy');paths=np.load(src/'paths.npz')
    g=[]
    for i in range(0,len(h),128):g.extend(10*np.log10(np.maximum(np.mean(abs(h[i:i+128].astype(np.complex128))**2,axis=(1,2)),1e-35)))
    g=np.array(g);p=z['position_m'];bs=np.array(cfg['bs_position_m']);np.savez_compressed(out/'plot_data.npz',time_s=t,gain_db=g,los=paths['los'],path_count=paths['count'],trajectory_time_s=z['time_s'],position_m=p,bs_position_m=bs)
    with (out/'gain.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['time_s','mean_abs_H_squared_db','los','path_count']);w.writerows(zip(t,g,paths['los'].astype(int),paths['count']))
    mapdir=Path('/root/autodl-tmp/Non-uniform-time-stamped-CSI/experiments/visual_occlusion_dataset_20260909');bg=Image.open(mapdir/'map_background.png');ext=np.load(mapdir/'plot_data.npz')['map_extent']
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained');ax.imshow(bg,extent=ext,origin='upper')
    pp=p[::10,:2];tt=z['time_s'][::10];segments=np.stack([pp[:-1],pp[1:]],axis=1);lc=LineCollection(segments,cmap='viridis',norm=plt.Normalize(0,t[-1]),linewidth=3);lc.set_array(tt[:-1]);ax.add_collection(lc)
    ax.scatter(*bs[:2],c='red',marker='*',edgecolors='black',s=190,zorder=7);ax.annotate(f'BS ({bs[0]:.2f}, {bs[1]:.2f})',bs[:2],xytext=(10,-16),textcoords='offset points')
    for label,point,off in [('Start (-53, 0)',p[0],(-12,12)),('End (-40, -20)',p[-1],(12,4))]:
        ax.scatter(*point[:2],s=45,c='white',edgecolors='black',zorder=6);ax.annotate(label,point[:2],xytext=off,textcoords='offset points',ha='right' if label.startswith('Start') else 'left')
    for fraction in [.3,.7]:
        k=int(len(p)*fraction);ax.annotate('',p[k+35,:2],p[k,:2],arrowprops=dict(arrowstyle='->',color='black',lw=1.5))
    ax.set(xlim=(-62,-27),ylim=(-32,6),xlabel='Scene X / east (m)',ylabel='Scene Y / north (m)',title=f'site_03: curved route\nUAV z={cfg["height_m"]:.2f} m; BS z={bs[2]:.2f} m');ax.set_aspect('equal');fig.colorbar(lc,ax=ax,label='Time (s)',shrink=.7)
    for suffix in ['png','pdf']:fig.savefig(out/f'01_arc_top_view.{suffix}',dpi=200,facecolor='white')
    plt.close(fig);fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    ax.fill_between(t,0,1,where=~paths['los'],transform=ax.get_xaxis_transform(),color='.9',label='NLoS',zorder=-2);ax.plot(t,g,c='#0072B2',lw=1,label='Full CSI samples (500 Hz)')
    ax.set(xlim=(0,t[-1]),xlabel='Time (s)',ylabel='Mean |H|² (dB)',title=f'site_03 arc: {cfg["speed_mps"]:.2f} m/s, {t[-1]:.3f} s');ax.grid(axis='y',color='.9');ax.legend(loc='best')
    for suffix in ['png','pdf']:fig.savefig(out/f'02_csi_gain.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    switches=(t[np.flatnonzero(paths['los'][1:]!=paths['los'][:-1])+1]).tolist()
    summary=dict(frames=len(t),gain_range_db=[float(g.min()),float(g.max())],zero_path_frames=int(sum(paths['count']==0)),los_fraction=float(np.mean(paths['los'])),los_switch_times_s=switches,max_adjacent_gain_change_db=float(np.max(abs(np.diff(g)))))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
    for name in ['config.json','physical_config.json','collision_check.json','complete.json']:shutil.copyfile(src/name,out/name)
    shutil.copyfile(src/'run.py',out/'generate.py');shutil.copyfile(__file__,out/'plot.py')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.source,args.output)
