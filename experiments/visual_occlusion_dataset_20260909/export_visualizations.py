"""Export compact, reproducible dataset inspection figures from final raw data."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':False,'pdf.fonttype':42})
def savefig(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def render_background(data,out,rows):
    import mitsuba as mi
    mi.set_variant('scalar_rgb');cfg=json.loads((data/'dataset_config.json').read_text());scene=mi.load_file(cfg['scene_path'])
    extent=np.array([-185.,100.,-125.,125.]);width,height=2280,2000
    im=Image.new('RGB',(width,height),(248,248,245));draw=ImageDraw.Draw(im);polys=[];zs=[]
    for mesh in scene.shapes():
        p=mi.traverse(mesh);v=np.asarray(p['vertex_positions']).reshape(-1,3);f=np.asarray(p['faces']).reshape(-1,3);tri=v[f]
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);cent=tri.mean(1)
        mask=(abs(cross[:,2])>.02)&(cent[:,2]>1)&(cent[:,0]>extent[0]-10)&(cent[:,0]<extent[1]+10)&(cent[:,1]>extent[2]-10)&(cent[:,1]<extent[3]+10)
        polys.append(tri[mask,:,:2]);zs.append(cent[mask,2])
    polys=np.concatenate(polys);zs=np.concatenate(zs)
    polys[:,:,0]=(polys[:,:,0]-extent[0])/(extent[1]-extent[0])*width
    polys[:,:,1]=(extent[3]-polys[:,:,1])/(extent[3]-extent[2])*height
    for idx in np.argsort(zs):
        shade=int(np.clip(226-zs[idx]*1.3,135,226));draw.polygon([tuple(p) for p in polys[idx]],fill=(shade,shade,shade))
    im.save(out/'map_background.png');return extent
def plot(data,out):
    rows=json.loads((data/'manifest.json').read_text());ext=render_background(data,out,rows);bg=Image.open(out/'map_background.png')
    poses=[];gains=[];times=[];los=[];bs=[]
    for r in rows:
        z=np.load(data/'routes'/r['traj_id']/'trajectory.npz');poses.append(z['position_m'][::50]);bs.append(r['bs_position_m'])
        rf=data/'csi'/r['traj_id'];h=np.load(rf/'H.npy',mmap_mode='r');parts=[]
        for i in range(0,len(h),128):parts.extend(10*np.log10(np.maximum(np.mean(abs(h[i:i+128].astype(np.complex128))**2,axis=(1,2)),1e-35)))
        gains.append(parts);times.append(np.load(rf/'time_s.npy'));los.append(np.load(rf/'paths.npz')['los'])
    poses=np.array(poses);bs=np.array(bs);gains=np.array(gains);times=np.array(times);los=np.array(los)
    np.savez_compressed(out/'plot_data.npz',route_ids=[r['traj_id'] for r in rows],position_m=poses,position_time_s=np.arange(81)/20,bs_position_m=bs,csi_time_s=times,gain_db=gains,los=los,map_extent=ext)
    with (out/'csi_gain_all_routes.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['route','time_s','mean_abs_H_squared_db','los'])
        for r,t,g,l in zip(rows,times,gains,los):w.writerows((r['traj_id'],float(tt),float(gg),int(ll)) for tt,gg,ll in zip(t,g,l))
    def background(ax):ax.imshow(bg,extent=ext,origin='upper');ax.set_aspect('equal');ax.set_xlabel('Scene X / east (m)');ax.set_ylabel('Scene Y / north (m)')
    fig,ax=plt.subplots(figsize=(11,9));background(ax)
    sitecolors=plt.get_cmap('tab10')
    for k in range(6):
        for i in range(k*4,k*4+4):ax.plot(poses[i,:,0],poses[i,:,1],color=sitecolors(k),lw=2)
        ax.scatter(*bs[k*4,:2],marker='*',s=150,c=[sitecolors(k)],edgecolors='black',zorder=5)
        ax.annotate(f'site_{k:02d}',bs[k*4,:2],xytext=(6,8),textcoords='offset points',weight='bold')
    ax.set_title('All 24 routes and six fixed base-station locations');savefig(fig,out,'01_map_overview')
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    for k,ax in enumerate(axes.flat):
        background(ax);group=poses[k*4:k*4+4];points=np.concatenate([group.reshape(-1,3),bs[k*4:k*4+1]])
        lo=points[:,:2].min(0)-7;hi=points[:,:2].max(0)+7;center=(lo+hi)/2;radius=max(hi-lo)/2
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius)
        for j in range(4):
            p=group[j];ax.plot(p[:,0],p[:,1],color=COLORS[j],lw=1.8,label=f'route_{j:02d}, z={p[0,2]:.1f} m')
            ax.scatter(*p[0,:2],s=24,facecolors='white',edgecolors=COLORS[j],zorder=6)
            ax.annotate('',p[45,:2],p[35,:2],arrowprops=dict(arrowstyle='->',color=COLORS[j],lw=1.8))
        ax.scatter(*bs[k*4,:2],marker='*',s=130,c='red',edgecolors='black',zorder=7)
        ax.set_title(f'site_{k:02d} ({rows[k*4]["split"]}), BS z={bs[k*4,2]:.1f} m');ax.legend(fontsize=8,loc='best')
    savefig(fig,out,'02_routes_by_site')
    fig,axes=plt.subplots(6,4,figsize=(17,18),layout='constrained')
    for i,ax in enumerate(axes.flat):
        ax.plot(times[i],gains[i],color=COLORS[i%4],lw=.85)
        ax.fill_between(times[i],0,1,where=~los[i],transform=ax.get_xaxis_transform(),color='.9',zorder=-2)
        ax.set_title(rows[i]['traj_id']);ax.set_xlim(0,4);ax.set_xticks(range(5));ax.grid(axis='y',color='.9',lw=.5)
        if i%4==0:ax.set_ylabel('Mean |H|² (dB)')
        if i>=20:ax.set_xlabel('Time (s)')
    savefig(fig,out,'03_csi_gain_all_routes')
    selections=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for k in range(6):
        r=rows[k*4];cap=data/'captures'/r['traj_id'];a=[v for v in json.loads((cap/'visibility_audit.json').read_text())['frames'] if v['camera']=='focus']
        visible=np.array([v['changed_pixels_in_target_roi']>=5 for v in a]);event=next((i for i in range(1,77) if visible[i:i+4].all() and not visible[i-1]),40)
        ids=sorted(set([0,max(0,event-2),min(80,event+2),min(80,event+10),80]))
        xy=np.array([v['projected_center_px'] for v in a]);x0=max(0,int(xy[:,0].min())-65);x1=min(1024,int(xy[:,0].max())+65);y0=max(0,int(xy[:,1].min())-70);y1=min(1024,int(xy[:,1].max())+70)
        tile=340;crop_h=max(1,round((y1-y0)*tile/(x1-x0)));canvas=Image.new('RGB',(tile*len(ids),tile+crop_h+100),'white');draw=ImageDraw.Draw(canvas)
        for j,idx in enumerate(ids):
            im=Image.open(cap/'rgb/focus'/f'{idx:04d}.png').convert('RGB');draw.text((j*tile+8,6),f'{idx/20:.2f} s',fill='black',font=font)
            canvas.paste(im.resize((tile,tile),Image.Resampling.LANCZOS),(j*tile,34));canvas.paste(im.crop((x0,y0,x1,y1)).resize((tile,crop_h),Image.Resampling.LANCZOS),(j*tile,tile+92))
        draw.text((8,tile+48),f'{r["traj_id"]} | full view above; identical fixed crop below',fill='black',font=font)
        canvas.save(out/f'04_rgb_site_{k:02d}.jpg',quality=94)
        selections.append(dict(route=r['traj_id'],camera='focus',frames=ids,times_s=[i/20 for i in ids],fixed_crop_xyxy=[x0,y0,x1,y1]))
    cap=data/'captures'/rows[0]['traj_id'];canvas=Image.new('RGB',(1600,435),'white');draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(['north','east','south','west']):
        draw.text((j*400+8,8),f'{name} | t=4.00 s',font=font,fill='black');canvas.paste(Image.open(cap/'rgb'/name/'0080.png').resize((400,400)),(j*400,35))
    canvas.save(out/'05_four_horizontal_views.jpg',quality=94)
    (out/'rgb_selection.json').write_text(json.dumps(selections,indent=2));(out/'manifest.json').write_text(json.dumps(rows,indent=2))
    if Path(__file__).resolve()!=(out/'export_visualizations.py').resolve():shutil.copyfile(__file__,out/'export_visualizations.py')
    print(out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'dataset');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);plot(args.dataset,args.output)
