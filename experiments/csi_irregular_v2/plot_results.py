#!/usr/bin/env python3
"""Render figures directly from saved result JSON; export the exact plot values."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#666666','#D55E00','#009E73','#222222','#0072B2','#CC79A7','#E69F00','#56B4E9']
MARKERS=['o','s','^','*','D','v','P','X']
LINESTYLES=['-','--',':','-.','-','--',':','-.']
ORDER=['hold','linear','kalman','harmonic','cru','latent_ode','neural_flow','tpatchgnn']
LABELS={'hold':'Persistence','linear':'Causal linear','kalman':'Kalman','cru':'CRU',
        'latent_ode':'Latent ODE','neural_flow':'Neural Flow','tpatchgnn':'t-PatchGNN','harmonic':'Harmonic LS'}


def collect(root):
    found={}
    for path in Path(root).glob('*/summary.json'):
        found[path.parent.name]=json.loads(path.read_text())
    return found


def model_name(name):return name.split('_seed')[0]


def grouped_points(found,group,metric='nmse_linear'):
    values=defaultdict(lambda:defaultdict(list))
    for name,result in found.items():
        for key,metrics in result['groups'].get(group,{}).items():
            values[model_name(name)][key].append(metrics[metric])
    return values


def export(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.png'),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_suite(root,out):
    from benchmark.reporting import condition_table,save_condition_table
    rows=condition_table(root)
    if not rows:return
    save_condition_table(rows,out/'all_conditions.csv')
    index={(r['condition'],r['model']):r for r in rows}
    offgrid=collect(Path(root)/'offgrid')
    if offgrid:
        values=grouped_points(offgrid,'query_band_ms')
        keys=sorted(set(k for model in values.values() for k in model),key=lambda k:float(k.split('-')[0]))
        fig,ax=plt.subplots(figsize=(7.16,3.1));plot_rows=[]
        for name in ORDER:
            if name not in values:continue
            kept=[(i,k) for i,k in enumerate(keys) if k in values[name]]
            y=[10*np.log10(max(np.mean(values[name][k]),1e-30)) for i,k in kept]
            idx=ORDER.index(name)
            ax.plot([i for i,k in kept],y,color=COLORS[idx],marker=MARKERS[idx],linestyle=LINESTYLES[idx],label=LABELS[name],linewidth=1.2,markersize=4)
            for (i,k),v in zip(kept,y):plot_rows.append({'model':name,'horizon_band_ms':k,'nmse_db':v,'nmse_linear':10**(v/10),'seeds':len(values[name][k])})
        ax.set_xticks(range(len(keys)),[k.replace('-','–') for k in keys]);ax.set_xlabel('Query horizon band (ms)')
        ax.set_ylabel('Full-matrix NMSE (dB)');ax.grid(axis='y',color='#ddd',linewidth=.5)
        ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.25))
        fig.tight_layout();export(fig,out/'true_offgrid_prediction')
        if plot_rows:
            with (out/'offgrid_plot_values.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(plot_rows[0]));w.writeheader();w.writerows(plot_rows)
    fig,axes=plt.subplots(2,2,figsize=(7.16,5.5))
    anydata=False
    for ax,pattern in zip(axes.flat,['uniform','random','bursty','dropout']):
        for model in ORDER:
            pts=[(n,index[(f'{pattern}_n{n}',model)]) for n in [4,8,16,32] if (f'{pattern}_n{n}',model) in index]
            if not pts:continue
            anydata=True;j=ORDER.index(model)
            ax.plot([n for n,r in pts],[r['nmse_db'] for n,r in pts],color=COLORS[j],marker=MARKERS[j],linestyle=LINESTYLES[j],label=LABELS[model],linewidth=1.2,markersize=4)
        ax.set_title(pattern.capitalize(),fontsize=9)
        ax.set_xlabel('History observations');ax.set_ylabel('NMSE (dB)');ax.set_xticks([4,8,16,32]);ax.grid(axis='y',color='#ddd',linewidth=.5)
    if anydata:
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='upper center',frameon=False)
        fig.tight_layout(rect=(0,0,1,.91));export(fig,out/'sampling_pattern_and_count')
    else:plt.close(fig)
    ablations=['timestamps_none','timestamps_uniform_times','timestamps_permuted_association','timestamps_jitter_5','timestamps_jitter_20']
    models=[m for m in ORDER if any((a,m) in index for a in ablations[1:])]
    if models:
        fig,ax=plt.subplots(figsize=(7.16,3.0))
        for m in models:
            x=[i for i,a in enumerate(ablations) if (a,m) in index]
            vals=[index[(ablations[i],m)] for i in x]
            ax.errorbar(x,[r['nmse_db'] for r in vals],yerr=[r['seed_nmse_db_std'] for r in vals],marker='o',capsize=3,label=LABELS[m])
        ax.set_xticks(range(5),['True times','Uniform labels','Shuffled pairs','5 ms noise','20 ms noise'])
        ax.set_ylabel('NMSE (dB)');ax.grid(axis='y',color='#ddd',linewidth=.5);ax.legend(frameon=False)
        fig.tight_layout();export(fig,out/'timestamp_ablation')
    uniform_models=[m for m in ORDER if ('uniform_trained_uniform',m) in index]
    if uniform_models:
        fig,ax=plt.subplots(figsize=(7.16,3.0))
        patterns=['uniform','random','bursty','dropout'];xx=np.arange(4)
        for m in uniform_models:
            for label,prefix,marker in [('Mixed training','', 'o'),('Uniform training','uniform_trained_', 's')]:
                keys=[(f'{p}_n16',m) if not prefix else (prefix+p,m) for p in patterns]
                yy=[index[k]['nmse_db'] if k in index else np.nan for k in keys]
                ax.plot(xx,yy,marker=marker,label=f'{LABELS[m]}: {label}')
        ax.set_xticks(xx,[p.capitalize() for p in patterns]);ax.set_ylabel('NMSE (dB)')
        ax.grid(axis='y',color='#ddd',linewidth=.5);ax.legend(frameon=False)
        fig.tight_layout();export(fig,out/'training_sampling_distribution')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    if (Path(a.input)/'main').is_dir():
        plot_suite(a.input,out)
        found=collect(Path(a.input)/'main')
    else:
        found=collect(a.input)
    if not found:raise SystemExit('No summary.json results found')
    raw=[]
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.0))
    # Main random-count subsets are not a controlled history-count experiment.
    # The count conclusion is reserved for the paired 16-condition stress plot.
    for ax,metric,ylabel in zip(axes,['nmse_linear','amplitude_nmse'],['Full-matrix NMSE (dB)','Amplitude NMSE (dB)']):
        group='query_s';xlabel='Prediction horizon (ms)'
        values=grouped_points(found,group,metric)
        for name in ORDER:
            if name not in values:continue
            keys=sorted(values[name],key=float)
            x=np.asarray([float(k) for k in keys])*(1000 if group=='query_s' else 1)
            mean=np.asarray([np.mean(values[name][k]) for k in keys])
            y=10*np.log10(np.maximum(mean,1e-30))
            idx=ORDER.index(name)
            ax.plot(x,y,color=COLORS[idx],marker=MARKERS[idx],linestyle=LINESTYLES[idx],linewidth=1.2,markersize=4,label=LABELS[name])
            for k,xv,yv,m in zip(keys,x,y,mean):
                raw.append({'figure':'prediction','group':group,'metric':metric,'model':name,'x':xv,'nmse_linear':m,'nmse_db':yv,'seeds':len(values[name][k])})
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(axis='y',color='#dddddd',linewidth=.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.04))
    fig.tight_layout(rect=(0,0,1,.89));export(fig,out/'prediction_horizon_complex_and_amplitude')
    for group in ['scenario','los_transition']:
        values=grouped_points(found,group)
        keys=sorted(set(k for g in values.values() for k in g))
        if len(keys)<2:continue
        fig,ax=plt.subplots(figsize=(7.16,3.1));names=[n for n in ORDER if n in values]
        width=.8/len(names);xx=np.arange(len(keys))
        for j,name in enumerate(names):
            yy=[10*np.log10(np.mean(values[name][k])) if k in values[name] else np.nan for k in keys]
            ax.bar(xx+(j-(len(names)-1)/2)*width,yy,width,color=COLORS[ORDER.index(name)],label=LABELS[name],edgecolor='black',linewidth=.3)
            for k,y in zip(keys,yy):raw.append({'figure':group,'group':group,'metric':'nmse_linear','model':name,'x':k,'nmse_linear':10**(y/10),'nmse_db':y,'seeds':len(values[name].get(k,[]))})
        ax.set_xticks(xx,keys,rotation=15 if group=='scenario' else 0,ha='center')
        ax.set_ylabel('Per-target NMSE (dB)');ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.set_axisbelow(True)
        ax.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.25))
        fig.tight_layout();export(fig,out/f'performance_by_{group}')
    with (out/'plot_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    (out/'aggregation.json').write_text(json.dumps({'source':str(Path(a.input).resolve()),
        'rule':'Mean per-target linear NMSE within each seed; mean across seeds; then 10log10. Bootstrap CIs are in per-model summary JSON, not seed error bars.'},indent=2))


if __name__=='__main__':main()
