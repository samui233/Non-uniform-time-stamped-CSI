"""Replot all six actual trajectory families from the bundled numeric assets."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['cruise','highspeed','sweep_turn','uturn','climb_descent','occlusion']
LABELS=['Cruise','High-speed pass','Sweeping turn','U-turn','Climb / descent','Occlusion crossings']


def main():
    with np.load(ROOT/'data/trajectory_map.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    xyz=d['xyz'];bs=d['bs_position_m']
    norm=Normalize(np.floor(xyz[:,2].min()),np.ceil(xyz[:,2].max()));cmap=plt.get_cmap('viridis')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
    gray=np.clip(.95-.0035*d['map_heights'],.71,.95)
    colors=np.c_[gray,gray,gray,np.ones(len(gray))]
    for k,(family,title,ax) in enumerate(zip(FAMILIES,LABELS,axes.flat)):
        ax.add_collection(PolyCollection(d['map_polygons'],facecolors=colors,edgecolors='none',rasterized=True,zorder=1))
        selected=np.flatnonzero(d['families']==family)
        for i in selected:
            p=xyz[d['offsets'][i]:d['offsets'][i+1]]
            segments=np.stack([p[:-1,:2],p[1:,:2]],1)
            collection=LineCollection(segments,cmap=cmap,norm=norm,linewidths=1.1,zorder=3)
            collection.set_array((p[:-1,2]+p[1:,2])/2);ax.add_collection(collection)
        ax.scatter(bs[0],bs[1],marker='*',s=110,c='black',edgecolors='white',linewidths=.7,zorder=5)
        ax.set(xlim=(-135,126),ylim=(-102,56),xlabel='Scene X (m)',ylabel='Scene Y (m)',
               title=f'({chr(97+k)}) {title} | {len(selected)} routes')
        ax.set_aspect('equal');ax.grid(False)
    colorbar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),shrink=.88,pad=.02)
    colorbar.set_label('UAV altitude: scene Z (m)')
    fig.legend(handles=[Line2D([],[],ls='',marker='*',color='black',ms=9,
        label=f'Fixed BS: ({bs[0]:g}, {bs[1]:g}, {bs[2]:g}) m; gray = city geometry')],
        loc='outside lower center',frameon=False)
    target=ROOT/'figures/00_trajectory_families_altitude'
    for ext in ('png','pdf'):fig.savefig(target.with_suffix('.'+ext),dpi=400,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
