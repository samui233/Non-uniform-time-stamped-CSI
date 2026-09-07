"""Replot published CSV values; missing v4 experiments are explicitly marked N/E."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
F=['cruise','highspeed','sweep_turn','uturn','climb_descent','occlusion']
T=['Cruise','High speed','Sweep turn','U-turn','Climb/descent','Occlusion v4']
M=['cru','latent_ode','neural_flow','tpatchgnn','time_rwkv_csi','ct_transformer_csi']
L=['CRU','Latent ODE','Flow','t-PatchGNN','Time-RWKV','CT-Transformer']
B=['hold','linear','kalman','harmonic'];C=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#333333']
def read(n):return list(csv.DictReader((ROOT/'tables'/n).open()))
R={(r['family'],r['case'],r['model']):r for r in read('overall.csv')}
def get(f,c,m,k='nmse_db'):return float(R.get((f,c,m),{}).get(k,'nan'))
def save(fig,n):
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/(n+'.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
def heat(ax,z,x,y,title,cmap='viridis',vmin=None,vmax=None):
    z=np.asarray(z,float);cm=plt.get_cmap(cmap).copy();cm.set_bad('#eeeeee');im=ax.imshow(np.ma.masked_invalid(z),aspect='auto',cmap=cm,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(x)),x,rotation=35,ha='right');ax.set_yticks(range(len(y)),y);ax.set_title(title,fontsize=10)
    for i in range(len(y)):
        for j in range(len(x)):
            val=z[i,j];rgba=im.cmap(im.norm(val));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{val:.2f}' if np.isfinite(val) else 'N/E',ha='center',va='center',fontsize=7,color='white' if np.isfinite(val) and lum<.5 else 'black')
    return im
def matrix(name,z,x,title):
    fig,ax=plt.subplots(figsize=(11,4.7));im=heat(ax,z,x,T,title);fig.colorbar(im,ax=ax);fig.tight_layout();save(fig,name)
def main():
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    matrix('01_overall_nmse',[[get(f,'main',m) for m in M+B] for f in F],L+['Hold','Linear','Kalman','Harmonic'],'Test NMSE (dB)')
    horizons=read('by_horizon.csv')
    for name,kind in [('02_prediction_horizon','time'),('03_observation_budget','count')]:
        fig,axes=plt.subplots(2,3,figsize=(11,6.2))
        for ax,f,title in zip(axes.flat,F,T):
            for m,label,color,mark in zip(M+['hold'],L+['Hold'],C,['o','s','^','D','v','P','x']):
                if kind=='time':
                    rows=sorted([r for r in horizons if r['family']==f and r['model']==m and r['case']=='main'],key=lambda r:float(r['query_s']));x=[1000*float(r['query_s']) for r in rows];y=[float(r['nmse_db']) for r in rows]
                else:x=[4,8,16,32];y=[get(f,c,m) for c in ['count_random_n4','count_random_n8','pattern_random_n16','count_random_n32']]
                ax.plot(x,y,label=label,color=color,marker=mark,ms=3,lw=1.1)
            ax.set_xscale('log');ax.set_xticks(x,[f'{v:g}' for v in x]);ax.set(title=title,ylabel='NMSE (dB)',xlabel='Prediction horizon (ms)' if kind=='time' else 'Historical observations');ax.grid(axis='y',color='.9')
            if kind=='count' and f=='occlusion':ax.text(.03,.04,'8 observations: N/E',transform=ax.transAxes,fontsize=7)
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=7,frameon=False);fig.tight_layout(rect=(0,0,1,.94));save(fig,name)
    for name,cases,labels,base in [('04_sampling_pattern',['pattern_random_n16','pattern_bursty_n16','pattern_dropout_n16'],['Random','Bursty','Gap'],'pattern_uniform_n16'),('05_timestamp_and_staleness',['timestamps_uniform_times','timestamps_jitter_20','stale_50','stale_100'],['Fake uniform','Jitter 20 ms','Stale 50 ms','Stale 100 ms'],'timestamps_none'),('06_history_information',['history_flat','history_reverse'],['Flat','Reverse'],'pattern_random_n16')]:
        zs=[np.array([[get(f,c,m)-get(f,base,m) for c in cases] for m in M]) for f in F];lim=max(.1,np.nanmax(np.abs(zs)))
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,z,title in zip(axes.flat,zs,T):im=heat(ax,z,labels,L,title,'RdBu_r',-lim,lim)
        fig.subplots_adjust(left=.13,right=.9,bottom=.15,hspace=.6,wspace=.65);fig.colorbar(im,cax=fig.add_axes([.94,.2,.015,.6]),label='NMSE change (dB); N/E = not evaluated');save(fig,name)
    matrix('07_offgrid_queries',[[get(f,'offgrid',m) for m in M+['hold']] for f in F],L+['Hold'],'Independent ray-traced off-grid NMSE (dB)')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,key,title in zip(axes,['gain_abs_error_db','shape_correlation'],['Gain MAE (dB)','Complex shape correlation']):
        im=heat(ax,[[get(f,'main',m,key) for m in M+['hold']] for f in F],L+['Hold'],T,title);fig.colorbar(im,ax=ax)
    fig.tight_layout();save(fig,'08_gain_and_shape')
    old={(r['family'],r['model']):float(r['delta_db']) for r in read('before_after.csv')}
    matrix('09_before_after',[[old.get((f,m),np.nan) for m in M[:4]] for f in F],L[:4],'Matched old/new NMSE change; v4 has no matched old run (N/E)')
    lat={(r['family'],r['model']):float(r['full_csi_wall_ms_median']) for r in read('latency.csv') if r['queries']=='1'}
    matrix('10_inference_latency',[[lat.get((f,m),np.nan) for m in M] for f in F],L,'Batch-1 full-CSI latency (ms); v4 not measured (N/E)')
if __name__=='__main__':main()
