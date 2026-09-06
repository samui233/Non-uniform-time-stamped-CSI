"""Read-only six-category experiment reporting from raw per-target CSVs.

No training, model selection, inference, target exclusion, or score clipping.
Example:
  python experiments/six_category_v1/report.py --run-root runs/six_category_v1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ['cruise', 'highspeed', 'sweep_turn', 'uturn', 'climb_descent', 'occlusion']
FAMILY_NAMES = dict(zip(FAMILIES, ['Cruise', 'High speed', 'Turning', 'U-turn', 'Climb/descent', 'Occlusion']))
MODELS = ['hold', 'linear', 'kalman', 'harmonic', 'cru', 'latent_ode', 'neural_flow',
          'tpatchgnn', 'observed_state_flow', 'local_dynamics']
LABELS = dict(zip(MODELS, ['Hold', 'Linear', 'Kalman', 'Harmonic', 'CRU', 'Latent ODE',
                          'Latent Flow', 't-PatchGNN', 'Observed Flow', 'Local dynamics']))
BASELINES = set(MODELS[:4])
ALIASES = {'observed_flow': 'observed_state_flow', 'local': 'local_dynamics',
           'local_dynamics_diagnostic': 'local_dynamics', 'latent_neural_flow': 'neural_flow'}
COLORS = ['#666666', '#009E73', '#E69F00', '#CC79A7', '#0072B2', '#D55E00',
          '#56B4E9', '#7A5195', '#003F5C', '#8B6F00']
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', '*']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '--', '-.']
IDENTITY = ['traj_id', 'sample_id', 'query_s']
METRICS = ['nmse', 'error', 'energy', 'amplitude_nmse', 'shape_correlation', 'gain_abs_error_db']
EXPECTED_SEEDS = {17, 29, 43}
STRESS_EXPERIMENTS = ['pattern_uniform_n16', 'pattern_random_n16', 'pattern_bursty_n16', 'pattern_dropout_n16',
                      'count_random_n4', 'count_random_n8', 'count_random_n32',
                      'timestamps_none', 'timestamps_uniform_times', 'timestamps_jitter_20', 'stale_50', 'stale_100']


def configure_plotting():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def db(value):
    return 10*np.log10(np.maximum(value, np.finfo(np.float64).tiny))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def parse_run(name):
    m = re.fullmatch(r'(.+)_seed(\d+)', name)
    model, seed = (m.group(1), int(m.group(2))) if m else (name, None)
    return ALIASES.get(model, model), seed


def canonical_family(name):
    return 'occlusion' if name in {'occlusion_v3', 'occlusion_new'} else name


def clean_rows(path):
    frame = pd.read_csv(path)
    required = set(IDENTITY + ['nmse', 'error', 'energy'])
    absent = required - set(frame)
    if absent:
        raise ValueError(f'{path}: missing required columns {sorted(absent)}')
    if frame.empty:
        raise ValueError(f'{path}: empty prediction table')
    frame['query_s'] = frame['query_s'].astype(float).round(9)
    for column in ['traj_id', 'sample_id', 'channel_group', 'event_category']:
        if column in frame:
            frame[column] = frame[column].fillna('unknown').astype(str)
    for column in ['los_transition', 'query_los']:
        if column in frame:
            frame[column] = frame[column].astype(str).str.lower().isin(['true', '1', '1.0'])
    for column in METRICS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
            if not np.isfinite(frame[column].to_numpy()).all():
                raise ValueError(f'{path}: nonfinite {column}; refusing to drop targets')
    if (frame['energy'] <= 0).any() or (frame['nmse'] < 0).any():
        raise ValueError(f'{path}: invalid target energy or NMSE; refusing silent filtering')
    if frame.duplicated(IDENTITY).any():
        raise ValueError(f'{path}: duplicate sample/query identity')
    if 'target_over_last_db' not in frame and 'last_observed_energy' in frame:
        energy = pd.to_numeric(frame['last_observed_energy'], errors='raise')
        if np.isfinite(energy).all() and (energy > 0).all():
            frame['target_over_last_db'] = 10*np.log10(frame['energy']/energy)
    if 'target_over_last_db' in frame:
        frame['target_over_last_db'] = pd.to_numeric(frame['target_over_last_db'], errors='raise')
    return frame


def combine_seeds(frames):
    """Average errors per identical target first; seeds are NOT new flights."""
    ordered = [f.sort_values(IDENTITY).reset_index(drop=True) for f in frames]
    reference = ordered[0]
    for f in ordered[1:]:
        if not reference[IDENTITY].equals(f[IDENTITY]):
            raise ValueError('Seed runs do not have identical target identities; cannot average them')
        for key in ['energy', 'channel_group', 'event_category', 'target_over_last_db']:
            if key in reference and key in f:
                if pd.api.types.is_numeric_dtype(reference[key]):
                    if not np.allclose(reference[key], f[key], rtol=2e-5, atol=1e-30, equal_nan=True):
                        raise ValueError(f'Seed runs disagree on target metadata {key}')
                elif not reference[key].equals(f[key]):
                    raise ValueError(f'Seed runs disagree on target metadata {key}')
    result = reference.copy()
    for key in METRICS:
        if all(key in f for f in ordered):
            result[key] = np.mean(np.stack([f[key].to_numpy() for f in ordered]), axis=0)
    return result


def cluster_key(frame, family):
    if family != 'occlusion':
        return 'traj_id'
    if 'channel_group' not in frame or frame['channel_group'].isin(['unknown', '', 'nan', 'None']).any():
        return None
    return 'channel_group'


def cluster_distribution(frame, family, repeats=1000):
    key = cluster_key(frame, family)
    if key is None:
        return None, 0, 'missing_channel_group_no_route_fallback'
    g = frame.groupby(key, sort=True)['nmse'].agg(['sum', 'count'])
    count = len(g)
    if count < 2:
        return None, count, 'fewer_than_two_independent_groups'
    rng = np.random.default_rng(713)
    ix = rng.integers(count, size=(repeats, count))
    sums, counts = g['sum'].to_numpy(), g['count'].to_numpy()
    values = sums[ix].sum(1)/counts[ix].sum(1)
    return values, count, 'channel_group' if family == 'occlusion' else 'trajectory'


def score(frame, family, repeats=1000):
    value = float(frame['nmse'].mean())
    distribution, count, status = cluster_distribution(frame, family, repeats)
    ci = np.quantile(db(distribution), [.025, .975]) if distribution is not None else [np.nan, np.nan]
    row = {'targets': len(frame), 'windows': frame[IDENTITY[:2]].drop_duplicates().shape[0],
           'nmse_linear': value, 'nmse_db': float(db(value)),
           'energy_weighted_nmse_db': float(db(frame['error'].sum()/frame['energy'].sum())),
           'median_target_nmse_db': float(np.median(db(frame['nmse'].to_numpy()))),
           'p90_target_nmse_db': float(np.quantile(db(frame['nmse'].to_numpy()), .9)),
           'fraction_target_nmse_below_0db': float((frame['nmse'] < 1).mean()),
           'cluster_count': count, 'cluster_kind': status,
           'nmse_db_ci_low': float(ci[0]), 'nmse_db_ci_high': float(ci[1])}
    for key in ['amplitude_nmse', 'shape_correlation', 'gain_abs_error_db']:
        if key in frame:
            row[key] = float(frame[key].mean())
    return row


def paired_score(a, b, family, repeats=1000):
    a = a.sort_values(IDENTITY).reset_index(drop=True)
    b = b.sort_values(IDENTITY).reset_index(drop=True)
    if not a[IDENTITY].equals(b[IDENTITY]):
        return {'pair_status': 'unmatched_target_sets_not_compared', 'paired_targets': 0,
                'delta_nmse_db': np.nan, 'delta_ci_low': np.nan, 'delta_ci_high': np.nan}
    if not np.allclose(a['energy'], b['energy'], rtol=2e-5, atol=1e-30):
        raise ValueError('Matched sample IDs have different physical target energies')
    key = cluster_key(a, family)
    difference = float(db(a['nmse'].mean()/b['nmse'].mean()))
    if key is None:
        return {'pair_status': 'matched_missing_channel_group_no_ci', 'paired_targets': len(a),
                'delta_nmse_db': difference, 'delta_ci_low': np.nan, 'delta_ci_high': np.nan}
    frame = pd.DataFrame({'cluster': a[key], 'a': a['nmse'], 'b': b['nmse']})
    g = frame.groupby('cluster')[['a', 'b']].sum()
    rng = np.random.default_rng(714)
    if len(g) >= 2:
        ix = rng.integers(len(g), size=(repeats, len(g)))
        values = db(g['a'].to_numpy()[ix].sum(1)/g['b'].to_numpy()[ix].sum(1))
        ci = np.quantile(values, [.025, .975])
    else:
        ci = [np.nan, np.nan]
    return {'pair_status': 'matched', 'paired_targets': len(a), 'paired_clusters': len(g),
            'delta_nmse_db': difference, 'delta_ci_low': float(ci[0]), 'delta_ci_high': float(ci[1])}


class Report:
    def __init__(self, run_root, output, repeats=1000, dpi=600):
        self.run_root, self.output = Path(run_root), Path(output)
        self.repeats, self.dpi = repeats, dpi
        self.groups = {}
        self.meta = {}
        self.sources, self.notes = [], []
        self.tables = {}
        self.figures = []
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'figures').mkdir(exist_ok=True)
        (self.output/'tables').mkdir(exist_ok=True)

    def load(self):
        grouped = defaultdict(list)
        per_run = []
        for path in sorted((self.run_root/'results').glob('*/*/*/per_target.csv')):
            if not path.with_name('summary.json').exists():
                self.notes.append(f'结果尚未完成保存，跳过缺少summary.json的CSV：{path}')
                continue
            experiment, family, run = path.relative_to(self.run_root/'results').parts[:3]
            family = canonical_family(family)
            model, seed = parse_run(run)
            if family not in FAMILIES or model not in MODELS:
                self.notes.append(f'未识别运行未纳入预设比较：{experiment}/{family}/{run}')
                continue
            frame = clean_rows(path)
            grouped[(experiment, family, model)].append((seed, frame))
            per_run.append({'experiment': experiment, 'family': family, 'model': model, 'seed': seed,
                            **score(frame, family, self.repeats)})
            self.sources.append({'path': str(path.resolve()), 'sha256': file_hash(path), 'targets': len(frame)})
        for key, values in grouped.items():
            seeds = [seed for seed, _ in values]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f'Duplicate canonical model aliases/seeds in {key}')
            self.groups[key] = combine_seeds([frame for _, frame in values])
            main_like = key[0] in {'main', 'offgrid'}
            expected = EXPECTED_SEEDS if key[2] not in BASELINES and main_like else ({None} if key[2] in BASELINES else {17})
            scores = [float(db(f['nmse'].mean())) for _, f in values]
            self.meta[key] = {'seed_count': len(seeds), 'seeds': ','.join(map(str, sorted(s for s in seeds if s is not None))),
                              'expected_seed_count': len(expected), 'seeds_complete': set(seeds) == expected,
                              'seed_nmse_db_min': min(scores), 'seed_nmse_db_max': max(scores)}
        self.tables['per_run'] = pd.DataFrame(per_run)

    def build_tables(self):
        overall, horizon, bands, occlusion, paired = [], [], [], [], []
        for key, frame in sorted(self.groups.items()):
            experiment, family, model = key
            context = {'experiment': experiment, 'family': family, 'model': model, **self.meta[key]}
            overall.append({**context, **score(frame, family, self.repeats)})
            if experiment != 'offgrid':
                for query, group in frame.groupby('query_s'):
                    horizon.append({**context, 'query_s': query, **score(group, family, self.repeats)})
            else:
                bucket = np.searchsorted([20, 50, 100, 200, 500], frame['query_s'].to_numpy()*1000-1e-5).clip(0, 4)
                labels = ['0–20', '20–50', '50–100', '100–200', '200–500']
                for ix in np.unique(bucket):
                    bands.append({**context, 'query_band_ms': labels[ix], **score(frame.loc[bucket == ix], family, self.repeats)})
            hold = self.groups.get((experiment, family, 'hold'))
            if hold is not None and model != 'hold':
                paired.append({**context, 'reference': 'hold', **paired_score(frame, hold, family, self.repeats)})
            if family == 'occlusion' and experiment == 'main':
                if 'event_category' in frame:
                    for category, group in frame.groupby('event_category'):
                        occlusion.append({**context, 'group_kind': 'event_category', 'group_value': category,
                                          'nmse_sum_fraction_of_method': float(group['nmse'].sum()/frame['nmse'].sum()),
                                          **score(group, family, self.repeats)})
                if 'los_transition' in frame:
                    for state, group in frame.groupby('los_transition'):
                        occlusion.append({**context, 'group_kind': 'future_los_transition', 'group_value': str(state),
                                          'nmse_sum_fraction_of_method': float(group['nmse'].sum()/frame['nmse'].sum()),
                                          **score(group, family, self.repeats)})
                if 'target_over_last_db' in frame:
                    values = frame['target_over_last_db'].to_numpy()
                    category = np.select([values <= -10, values <= -3, values < 3],
                                         ['Drop >=10 dB', 'Drop 3–10 dB', 'Change <3 dB'], default='Rise >=3 dB')
                    category[~np.isfinite(values)] = 'Unknown energy ratio'
                    for cat in dict.fromkeys(category):
                        subgroup = frame.loc[category == cat]
                        occlusion.append({**context, 'group_kind': 'target_change_vs_last', 'group_value': cat,
                                          'nmse_sum_fraction_of_method': float(subgroup['nmse'].sum()/frame['nmse'].sum()),
                                          **score(subgroup, family, self.repeats)})
        self.tables.update(overall=pd.DataFrame(overall), by_horizon=pd.DataFrame(horizon),
                           offgrid_horizon_bands=pd.DataFrame(bands), occlusion_groups=pd.DataFrame(occlusion),
                           paired_vs_hold=pd.DataFrame(paired))
        self.build_macro()
        self.build_timestamp_pairs()
        self.build_control_pairs()

    def build_control_pairs(self):
        rows = []
        for key, frame in self.groups.items():
            experiment, family, model = key
            if not (experiment.startswith(('pattern_', 'count_', 'stale_'))):
                continue
            if experiment == 'pattern_random_n16':
                continue
            reference = self.groups.get(('pattern_random_n16', family, model))
            if reference is None:
                self.notes.append(f'{experiment}/{family}/{model}缺少pattern_random_n16同窗参考。')
                continue
            rows.append({'experiment': experiment, 'family': family, 'model': model,
                         'reference_experiment': 'pattern_random_n16',
                         **paired_score(frame, reference, family, self.repeats)})
        self.tables['sampling_and_staleness_paired_changes'] = pd.DataFrame(rows)

    def build_macro(self):
        rows = []
        for experiment in ['main', 'offgrid']:
            for model in MODELS:
                keys = [(experiment, family, model) for family in FAMILIES]
                available = [key for key in keys if key in self.groups]
                if not available:
                    continue
                values = [self.groups[k]['nmse'].mean() for k in available]
                row = {'experiment': experiment, 'model': model, 'families_present': len(available),
                       'complete_six_families': len(available) == 6,
                       'all_seeds_complete': all(self.meta[k]['seeds_complete'] for k in available),
                       'macro_nmse_linear': float(np.mean(values)), 'macro_nmse_db': float(db(np.mean(values))),
                       'micro_nmse_db': float(db(pd.concat([self.groups[k]['nmse'] for k in available]).mean()))}
                distributions = [cluster_distribution(self.groups[k], k[1], self.repeats)[0] for k in available]
                # Per-family bootstrap samples are permuted independently before macro-averaging.
                if all(v is not None for v in distributions):
                    rng = np.random.default_rng(715)
                    vals = np.mean([v[rng.permutation(len(v))] for v in distributions], axis=0)
                    ci = np.quantile(db(vals), [.025, .975])
                else:
                    ci = [np.nan, np.nan]
                row.update(macro_ci_low=float(ci[0]), macro_ci_high=float(ci[1]))
                rows.append(row)
        self.tables['macro_category_scores'] = pd.DataFrame(rows)

    def build_timestamp_pairs(self):
        rows = []
        experiments = {key[0] for key in self.groups}
        clean_name = next((name for name in ['timestamps_none', 'timestamp_none', 'timestamps_clean'] if name in experiments), None)
        for key, frame in self.groups.items():
            experiment, family, model = key
            if 'timestamp' not in experiment or experiment == clean_name:
                continue
            clean = self.groups.get((clean_name, family, model)) if clean_name else None
            if clean is None:
                self.notes.append(f'{experiment}/{family}/{model}缺少同窗clean时间戳对照，不用main替代。')
                continue
            rows.append({'experiment': experiment, 'family': family, 'model': model, 'reference_experiment': clean_name,
                         **paired_score(frame, clean, family, self.repeats)})
        self.tables['timestamp_paired_changes'] = pd.DataFrame(rows)

    def save_tables(self):
        for name, table in self.tables.items():
            if not table.empty:
                table.to_csv(self.output/'tables'/f'{name}.csv', index=False)

    def finish_figure(self, fig, name):
        for ax in fig.axes:
            ax.set_facecolor('white')
        fig.savefig(self.output/'figures'/f'{name}.pdf', facecolor='white', bbox_inches='tight')
        fig.savefig(self.output/'figures'/f'{name}.png', dpi=self.dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        self.figures.append(name)

    @staticmethod
    def heatmap(ax, matrix, rowlabels, collabels, title, signed=False, limits=None):
        finite = matrix[np.isfinite(matrix)]
        if not len(finite):
            ax.text(.5, .5, 'Results not available', transform=ax.transAxes, ha='center', va='center')
            ax.set_axis_off()
            return None
        low, high = limits if limits is not None else (float(finite.min()), float(finite.max()))
        low, high = min(low, -1e-6), max(high, 1e-6)
        norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_xticks(range(len(collabels)), collabels, rotation=45, ha='right')
        ax.set_yticks(range(len(rowlabels)), rowlabels)
        ax.set_title(title, fontsize=10)
        ax.tick_params(length=0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value):
                    text, color = '—', '#777777'
                else:
                    text = f'{value:+.2f}' if signed else f'{value:.2f}'
                    color = 'white' if norm(value) < .16 or norm(value) > .84 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=7)
        return image

    def matrix(self, experiment, field='nmse_db'):
        result = np.full((len(FAMILIES), len(MODELS)), np.nan)
        table = self.tables['overall']
        if table.empty:
            return result
        for i, family in enumerate(FAMILIES):
            for j, model in enumerate(MODELS):
                rows = table[(table['experiment'] == experiment) & (table['family'] == family) & (table['model'] == model)]
                if len(rows):
                    if rows.iloc[0]['seeds_complete']:
                        result[i, j] = rows.iloc[0][field]
        return result

    def plot_main(self):
        matrix = self.matrix('main')
        if not np.isfinite(matrix).any():
            return
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        im = self.heatmap(ax, matrix, [FAMILY_NAMES[f] for f in FAMILIES], [LABELS[m] for m in MODELS],
                          'Full-matrix per-target NMSE (dB); lower is better')
        fig.colorbar(im, ax=ax, shrink=.85, label='NMSE (dB)')
        self.finish_figure(fig, 'main_by_category')
        table = self.tables['by_horizon']
        table = table[(table['experiment'] == 'main') & table['seeds_complete']]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, family in zip(axes.flat, FAMILIES):
            if not len(table[table['family'] == family]):
                ax.text(.5, .5, 'No completed runs', transform=ax.transAxes, ha='center', va='center')
            for j, model in enumerate(MODELS):
                rows = table[(table['family'] == family) & (table['model'] == model)].sort_values('query_s')
                if len(rows):
                    ax.plot(rows['query_s']*1000, rows['nmse_db'], label=LABELS[model], color=COLORS[j],
                            marker=MARKERS[j], linestyle=LINESTYLES[j], lw=1.1, markersize=3.5)
            ax.set_title(FAMILY_NAMES[family]); ax.set_xlabel('Prediction horizon (ms)'); ax.set_ylabel('NMSE (dB)')
            ax.set_xscale('log'); ax.set_xticks([10, 20, 50, 100, 200, 500], ['10', '20', '50', '100', '200', '500'])
            ax.grid(axis='y', color='.9', lw=.5)
        all_handles = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.update(dict(zip(labels, handles)))
        labels = [LABELS[m] for m in MODELS if LABELS[m] in all_handles]
        fig.legend([all_handles[label] for label in labels], labels, loc='outside upper center', ncol=5, frameon=False)
        self.finish_figure(fig, 'main_by_horizon')

    def plot_offgrid(self):
        matrices = [self.matrix(exp) for exp in ['main', 'offgrid']]
        if not np.isfinite(matrices[1]).any():
            return
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
        limits = (float(finite.min()), float(finite.max()))
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
        for ax, matrix, title in zip(axes, matrices, ['Dense-grid targets', 'Actual off-grid RT targets']):
            im = self.heatmap(ax, matrix, [FAMILY_NAMES[f] for f in FAMILIES], [LABELS[m] for m in MODELS], title, limits=limits)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=.85, label='NMSE (dB)')
        self.finish_figure(fig, 'offgrid_comparison')

    def plot_sampling(self):
        experiments = [[f'pattern_{pattern}_n16' for pattern in ['uniform', 'random', 'bursty', 'dropout']],
                       ['count_random_n4', 'count_random_n8', 'pattern_random_n16', 'count_random_n32']]
        matrices, labels = [], []
        for names in experiments:
            rows, rowlabels = [], []
            for i, family in enumerate(FAMILIES):
                for name in names:
                    rows.append(self.matrix(name)[i])
                    rowlabels.append(FAMILY_NAMES[family]+' / '+name.replace('pattern_', '').replace('count_', ''))
            matrices.append(np.array(rows)); labels.append(rowlabels)
        if not any(np.isfinite(m).any() for m in matrices):
            return
        fig, axes = plt.subplots(1, 2, figsize=(17, 10), constrained_layout=True)
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
        limits = (float(finite.min()), float(finite.max()))
        for ax, matrix, rowlabels, title in zip(axes, matrices, labels, ['Observation pattern (N=16)', 'Observation count (random times)']):
            im = self.heatmap(ax, matrix, rowlabels, [LABELS[m] for m in MODELS], title, limits=limits)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=.75, label='NMSE (dB)')
        self.finish_figure(fig, 'sampling_pattern_and_count')

    def plot_timestamps(self):
        table = self.tables['timestamp_paired_changes']
        if table.empty:
            return
        names = sorted(table['experiment'].unique())
        fig, axes = plt.subplots(1, len(names), figsize=(8*len(names), 4), squeeze=False, constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            matrix = np.full((6, 10), np.nan)
            for i, family in enumerate(FAMILIES):
                for j, model in enumerate(MODELS):
                    rows = table[(table['experiment'] == name) & (table['family'] == family) & (table['model'] == model)]
                    if len(rows): matrix[i, j] = rows.iloc[0]['delta_nmse_db']
            im = self.heatmap(ax, matrix, [FAMILY_NAMES[f] for f in FAMILIES], [LABELS[m] for m in MODELS], name, signed=True)
            if im is not None: fig.colorbar(im, ax=ax, shrink=.85, label='NMSE change vs same-window clean (dB)')
        self.finish_figure(fig, 'timestamp_sensitivity')

    def plot_staleness(self):
        table = self.tables['sampling_and_staleness_paired_changes']
        if table.empty:
            return
        table = table[table['experiment'].str.startswith('stale_')]
        if table.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
        for ax, experiment in zip(axes, ['stale_50', 'stale_100']):
            matrix = np.full((6, 10), np.nan)
            for i, family in enumerate(FAMILIES):
                for j, model in enumerate(MODELS):
                    rows = table[(table['experiment'] == experiment) & (table['family'] == family) & (table['model'] == model)]
                    if len(rows): matrix[i, j] = rows.iloc[0]['delta_nmse_db']
            im = self.heatmap(ax, matrix, [FAMILY_NAMES[f] for f in FAMILIES], [LABELS[m] for m in MODELS],
                              'Latest observation age = '+experiment.split('_')[1]+' ms', signed=True)
            if im is not None: fig.colorbar(im, ax=ax, shrink=.85, label='NMSE change vs same-target fresh history (dB)')
        self.finish_figure(fig, 'recent_observation_staleness')

    def plot_occlusion(self):
        table = self.tables['occlusion_groups']
        if table.empty:
            return
        table = table[table['seeds_complete']]
        if table.empty:
            return
        kinds = [k for k in ['event_category', 'future_los_transition', 'target_change_vs_last'] if k in set(table['group_kind'])]
        fig, axes = plt.subplots(len(kinds), 1, figsize=(11, 3.1*len(kinds)), squeeze=False, constrained_layout=True)
        for ax, kind in zip(axes.flat, kinds):
            sub = table[table['group_kind'] == kind]
            values = sorted(sub['group_value'].astype(str).unique())
            matrix = np.full((len(values), 10), np.nan)
            rowlabels = []
            for i, value in enumerate(values):
                block = sub[sub['group_value'].astype(str) == value]
                friendly = {'history_and_future_switch': 'History + future switch', 'history_switch_only': 'History switch only',
                            'future_switch_only': 'Future switch only', 'stable_los': 'Stable LoS', 'stable_nlos': 'Stable NLoS',
                            'True': 'Future LoS state changes', 'False': 'No future LoS state change'}.get(value, value)
                rowlabels.append(friendly+f" (n={int(block.iloc[0]['targets'])})")
                for j, model in enumerate(MODELS):
                    row = block[block['model'] == model]
                    if len(row): matrix[i, j] = row.iloc[0]['nmse_db']
            title = {'event_category': 'History/future switching events', 'future_los_transition': 'Future LoS/NLoS transitions',
                     'target_change_vs_last': 'True target energy relative to latest observation'}.get(kind, kind)
            im = self.heatmap(ax, matrix, rowlabels, [LABELS[m] for m in MODELS], title)
            if im is not None: fig.colorbar(im, ax=ax, shrink=.85, label='NMSE (dB)')
        self.finish_figure(fig, 'occlusion_event_and_fading')

    def completeness(self):
        missing = []
        for experiment in ['main', 'offgrid']:
            for family in FAMILIES:
                for model in MODELS:
                    key = (experiment, family, model)
                    if key not in self.groups:
                        missing.append('/'.join(key)+': missing')
                    elif not self.meta[key]['seeds_complete']:
                        missing.append('/'.join(key)+': incomplete seeds '+self.meta[key]['seeds'])
        return missing

    def stress_completeness(self):
        missing = []
        for experiment in STRESS_EXPERIMENTS:
            for family in FAMILIES:
                for model in MODELS:
                    key = (experiment, family, model)
                    if key not in self.groups:
                        missing.append('/'.join(key)+': missing')
                    elif not self.meta[key]['seeds_complete']:
                        missing.append('/'.join(key)+': unexpected seeds '+self.meta[key]['seeds'])
        return missing

    def describe_pairs(self, table, prefix=None):
        """Describe observed deltas; 0.01 dB is display tolerance, not a test."""
        lines = []
        if table.empty:
            return lines
        if prefix is not None:
            table = table[table['experiment'].str.startswith(prefix)]
        for experiment, block in table.groupby('experiment'):
            block = block[np.isfinite(block['delta_nmse_db'])]
            if block.empty:
                continue
            values = block['delta_nmse_db']
            worse, better = int((values > .01).sum()), int((values < -.01).sum())
            similar = len(values)-worse-better
            lines += ['', f'{experiment}：在已严格配对的 {len(values)} 个类别—方法条件中，'
                      f'{worse} 个均值退化超过0.01dB，{better} 个改善超过0.01dB，{similar} 个差异在±0.01dB内；'
                      f'完整差值范围为 {values.min():+.3f} 至 {values.max():+.3f}dB。'
                      '0.01dB只是显示精度的描述界限，不是统计显著性检验，配对区间保留在CSV。']
        return lines

    def write_report(self):
        missing = self.completeness()
        text = ['# 六类场景的非均匀时间戳 CSI 预测实验', '',
                '本报告从保存的逐目标 CSV 重新计算；不重新训练或推理，不删除深衰落，不平滑指标，不裁剪正 NMSE。', '',
                '## 实验问题与比较边界', '',
                '六类分别训练，用于研究各运动与传播条件下单模态 CSI 的可预测性。类别并非互斥物理因素：前五类主要描述运动，遮挡类描述传播条件，不能将类别间全部差异归因于某个单一因素。', '',
                'CRU、Latent ODE、Latent Neural Flow 和 t-PatchGNN 是原方法面向矩阵 CSI 的适配；Observed Flow 是保留已观测 CSI 状态的修复适配；Local dynamics 是另行设计的轻量局部动态诊断。它们的编码器、潜变量维数、状态锚点和输出结构不完全相同，因此不是只替换连续时间核心的严格架构消融，更不能把后两者冒充原论文方法。', '',
                '所有方法只能读取历史 CSI、历史时间戳及查询时刻。LoS、通道组和衰落标签仅用于结果分组。前五类来自原分类数据，第六类来自新空间通道划分的遮挡数据；数据配置与checkpoint以各运行保存的配置为准。', '',
                '预测对象是以收发阵列中心几何距离为传播参考的完整复数CSI，不是保留未知绝对载波相位的原始CSI。所有方法使用相同参考定义；模型不输入未来位置，也不在评价时对每个预测做oracle相位对齐。恢复绝对传播参考需要另外知道对应几何时延，不能把这里的结果直接解释为已预测该绝对相位。', '',
                '前五类复用原训练集拟合的128维复数PCA（256维实数输入）；新遮挡类使用新TRAIN拟合的256维复数PCA（512维实数输入）。后者是基于TRAIN/VAL重建精度选择，不使用TEST选维。每个数据集内部的学习方法共享同一PCA，但跨类别输入维度并不完全相同。', '',
                '每类每轮4000个TRAIN窗口，VAL固定600窗口，最多60轮。CRU批量512；Latent ODE和Latent Flow为256；t-PatchGNN前五类64、新遮挡类16；Observed Flow为128；Local dynamics为256。因此每轮优化更新次数及计算开销不同，不能称为只改变模型核心的严格等预算消融。Neural Flow早停耐心10，其余12。部分同数据、同实现、同配置的已完成权重通过REUSED.json复用，不把复用模型记为重新训练。', '',
                '## 指标与统计口径', '',
                '主指标先对每个目标计算完整复数矩阵误差与该目标能量的比值，在线性域等权平均后转 dB；不是逐目标 dB 的算术平均。另列能量加权误差、中位数、90分位和低于0 dB的比例。负 NMSE 只是优于零输出，不等于优于保持。', '',
                '同一模型三个种子先在相同目标上平均线性误差，再计算总体分数。种子范围另行列出，种子不作为新增独立轨迹。95%区间按独立轨迹聚类重采样；新遮挡类按 channel_group 重采样，不能把同一通道的三条变体当作独立组。若缺少通道ID，该类区间标记不可用，不回退成逐轨迹区间。每类独立组很少，区间精度有限。', '',
                '与保持法或干净时间戳的差值只在样本ID、轨迹和查询时刻完全一致时计算。负差值代表更好；未匹配窗口不强行求配对差。跨类宏平均先对每类线性NMSE求均值，避免窗口数更多的类别自动占据更大权重。', '']
        text += ['## 完成状态', '', f'已读取 {len(self.sources)} 个逐目标结果文件、形成 {len(self.groups)} 个实验/类别/方法汇总。']
        if missing:
            text += ['', f'**尚有 {len(missing)} 项主实验或离网格实验缺失/种子未齐。当前为阶段性汇总，不能当作全部实验已完成。**',
                     '完整缺失列表见 `report_manifest.json`。图中只绘制预定种子已齐的方法条件，缺失或种子未齐项为破折号；原始部分结果仍完整列于CSV，不据此最终挑选方法。']
        else:
            text += ['', '主实验与真实离网格查询的全部预定方法/种子已存在。受控条件的可用运行另见原始汇总表。']
        missing_stress = self.stress_completeness()
        text += ['', f'受控采样/时间戳/近期观测缺失实验尚缺 {len(missing_stress)} 个方法条件；完整列表同样保存在manifest。']
        overall = self.tables['overall']
        paired = self.tables['paired_vs_hold']
        text += ['', '## 不同场景与预测提前量', '',
                 '主实验使用每类固定测试窗口及六个未来查询提前量。首先比较完整目标上的误差，再查看差异是否集中在某个预测距离。跨种子均值不能掩盖某个种子近零输出或训练失效。']
        if not overall.empty:
            text += ['', '| 类别 | 方法 | 种子数 | NMSE/dB | 95%组区间/dB | 相对保持/dB |', '|---|---|---:|---:|---|---:|']
            for _, row in overall[overall['experiment'] == 'main'].iterrows():
                delta = np.nan
                if not paired.empty:
                    hit = paired[(paired['experiment'] == 'main') & (paired['family'] == row['family']) & (paired['model'] == row['model'])]
                    if len(hit): delta = hit.iloc[0]['delta_nmse_db']
                interval = f"[{row['nmse_db_ci_low']:.2f}, {row['nmse_db_ci_high']:.2f}]" if np.isfinite(row['nmse_db_ci_low']) else '不可用'
                delta_text = f'{delta:+.3f}' if np.isfinite(delta) else '—'
                flag = '' if row['seeds_complete'] else '（未齐）'
                text.append(f"| {FAMILY_NAMES[row['family']]} | {LABELS[row['model']]} | {row['seed_count']}{flag} | {row['nmse_db']:.3f} | {interval} | {delta_text} |")
        for name in ['main_by_category', 'main_by_horizon']:
            if name in self.figures: text += ['', f'![{name}](figures/{name}.png)']
        if not paired.empty:
            valid = paired[(paired['experiment'] == 'main') & paired['seeds_complete'] & paired['delta_nmse_db'].notna()]
            for family in FAMILIES:
                wins = valid[(valid['family'] == family) & (valid['delta_nmse_db'] < 0)]
                neural = [LABELS[m] for m in wins['model'] if m not in BASELINES]
                any_rows = valid[valid['family'] == family]
                if len(any_rows):
                    text += ['', f"{FAMILY_NAMES[family]}：已齐种子的学习方法中，跨种子均值低于同窗保持的有：{('、'.join(neural)) if neural else '暂无'}。这是观察到的均值比较，不自动代表统计显著或每个提前量、每条轨迹均胜出。"]
        horizon = self.tables['by_horizon']
        if not horizon.empty:
            main_h = horizon[(horizon['experiment'] == 'main') & horizon['seeds_complete']]
            for model in MODELS[4:]:
                model_rows = main_h[main_h['model'] == model]
                hold_rows = main_h[main_h['model'] == 'hold']
                comparison = model_rows.merge(hold_rows, on=['family', 'query_s'], suffixes=('_model', '_hold'))
                if len(comparison):
                    wins = int((comparison['nmse_linear_model'] < comparison['nmse_linear_hold']).sum())
                    text += ['', f'{LABELS[model]}：在已齐种子且有保持对照的 {len(comparison)} 个类别—提前量条件中，'
                             f'{wins} 个均值优于保持。全部提前量均纳入这一计数，不只保留有利区间。']
        text += ['', '## 真实离网格未来查询', '',
                 '离网格目标来自真实射线追踪，不是相邻CSI线性插值得到的标签。图中并列给出主实验和离网格结果；两者查询提前量分布与截止窗口可能不同，因此跨图分数变化不能单独归因为“非网格时间更难”。实际连续查询按提前量区间的完整统计见 `tables/offgrid_horizon_bands.csv`。']
        if 'offgrid_comparison' in self.figures: text += ['', '![真实离网格查询](figures/offgrid_comparison.png)']
        if not overall.empty:
            off = overall[(overall['experiment'] == 'offgrid') & overall['seeds_complete']]
            for family, block in off.groupby('family'):
                neural = block[~block['model'].isin(BASELINES)]
                if len(neural):
                    below = int((neural['nmse_db'] < 0).sum())
                    text += ['', f"{FAMILY_NAMES[family]}的已齐离网格学习方法中，{below}/{len(neural)} 个总体NMSE为负，"
                             f"分数范围为 {neural['nmse_db'].min():.3f} 至 {neural['nmse_db'].max():.3f}dB。"
                             '这描述实际连续查询标签上的结果，不把它与密集网格均值的差解释为单一采样因素。']
        text += ['', '## 不规则历史的数量与分布', '',
                 '受控实验固定种子17，每个条件使用相同目标窗口；分别改变16个历史观测的时间分布，或在随机分布下改变观测数量。各条件均保持相同历史跨度、未来查询范围和端点约束。均匀采样在密集离散网格上取整，并非所有数量下都严格等间隔。不能拿200窗口的受控实验与600窗口主实验直接相减解释收益。']
        if 'sampling_pattern_and_count' in self.figures: text += ['', '![历史采样控制](figures/sampling_pattern_and_count.png)']
        text += self.describe_pairs(self.tables['sampling_and_staleness_paired_changes'], prefix=('pattern_', 'count_'))
        text += ['', '## 报告时间戳不准确的影响', '',
                 '时间戳扰动不改变未来标签；仅调整内部历史时间，首尾时间固定，CSI及完整H随被扰动时间戳同步重排。统一内部时间标签或20ms扰动与同窗干净标签比较，保持法因最后观测不变而应保持原分数。该设置未覆盖全局传感器时钟偏移、漂移或最新观测本身的时间误差。图中为扰动后的NMSE减干净条件NMSE，正值代表退化。缺少完全匹配的clean运行时不计算差值。具体扰动规则以评估代码为准。']
        if 'timestamp_sensitivity' in self.figures: text += ['', '![时间戳敏感性](figures/timestamp_sensitivity.png)']
        timestamps = self.tables['timestamp_paired_changes']
        text += self.describe_pairs(timestamps)
        if not timestamps.empty:
            hold_change = timestamps[(timestamps['model'] == 'hold') & (timestamps['delta_nmse_db'].abs() > 1e-5)]
            if len(hold_change):
                message = '注意：报告发现保持法在内部时间戳扰动下出现非零配对差，需检查评估契约；原始数值未被改成零。'
                self.notes.append(message)
                text += ['', '**'+message+'**']
        text += ['', '## 近期观测缺失', '',
                 'stale_50和stale_100保持16个真实随机历史观测，但最近一帧分别停在截止点前50或100ms；最早历史仍约为−0.4s。目标时刻保持不变，按实际可用历史重新归一化，与pattern_random_n16的相同目标配对比较。', '',
                 '这一实验同时改变历史覆盖范围、最近观测年龄，以及从最近CSI到预测目标的有效外推距离，反映更新延迟/近期观测缺失，不能解释成单纯时间戳误差或只改变内部采样间隔。其保持法也必须使用实际可用的最后一帧，不能借用已缺失的新观测。']
        if 'recent_observation_staleness' in self.figures: text += ['', '![近期观测缺失](figures/recent_observation_staleness.png)']
        text += self.describe_pairs(self.tables['sampling_and_staleness_paired_changes'], prefix='stale_')
        text += ['', '## 遮挡切换与深衰落', '',
                 'LoS/NLoS切换不等于增益一定显著变化，持续NLoS中也可能因反射路径或多径相消产生深衰落。因此同时按事件类别、未来区间是否发生LoS状态变化，以及真实目标能量相对最后观测能量的变化分组。', '',
                 '深衰落描述阈值预设为相对最近观测下降至少10dB，另列下降3–10dB、变化小于3dB和上升至少3dB。阈值只用于描述，不删除样本、不修改损失、不根据模型误差定义“异常”。如果CSV没有真实最后观测能量或其比值，不能用预测增益误差替代这一物理标签。']
        if 'occlusion_event_and_fading' in self.figures: text += ['', '![遮挡和衰落分组](figures/occlusion_event_and_fading.png)']
        occlusion = self.tables['occlusion_groups']
        if not occlusion.empty:
            deep = occlusion[(occlusion['group_kind'] == 'target_change_vs_last') &
                             (occlusion['group_value'] == 'Drop >=10 dB') & occlusion['seeds_complete']]
            if len(deep):
                text += ['', f"当前下降至少10dB分组包含 {int(deep.iloc[0]['targets'])} 个真实目标，"
                         '这些目标全部保留于总体评价。不同模型在该组的误差占各自总逐目标NMSE和的比例为：']
                text += ['', '| 方法 | 该组NMSE/dB | 占该方法总NMSE和/% | 独立通道数 |', '|---|---:|---:|---:|']
                for _, row in deep.iterrows():
                    text.append(f"| {LABELS[row['model']]} | {row['nmse_db']:.3f} | {100*row['nmse_sum_fraction_of_method']:.2f} | {row['cluster_count']} |")
                text += ['', '这个占比用于判断相对误差是否被深衰落主导；占比高并不意味着应该删掉该组，也不能仅凭压低输出后的总体均值就认定恢复了更多信道结构。']
        text += ['', '## 可以与不可以据此得出的结论', '',
                 '只有在完整同窗目标上比保持更好，且跨种子与独立通道的结果一致时，才能较有把握地说方法利用了可预测动态。单纯NMSE负值、压低输出幅度改善深衰落尾部、或只展示最好种子，都不足以证明模型恢复了CSI结构。', '',
                 '本实验只研究CSI单模态。某个遮挡组预测困难可以定位未来多模态研究的条件，但不能直接证明图像一定有效；图像是否提供增量信息仍需独立多模态实验。若这些测试结果用于进一步改模型，应另留新的最终测试，不能把反复开发使用的集合称为盲测。', '',
                 '所有图均从表中精确值绘制，保留正NMSE和差模型，输出白底600dpi PNG及嵌入TrueType字体的PDF。置信区间、种子范围及原始来源哈希保留在CSV/JSON，而非靠图中近似读数。', '',
                 '## 文件', '', '- `tables/`：逐运行、跨种子、按提前量、配对差、遮挡组及跨类宏平均。',
                 '- `figures/`：PNG和PDF。', '- `report_manifest.json`：源码哈希、全部原始CSV哈希、缺失清单与警告。']
        text += ['', '## 从公开汇总表重新绘图', '',
                 '公开文件夹只需保留本说明、tables/、figures/及scripts/中的report.py和replay_figures.py。重画仅依赖numpy、pandas、matplotlib，不需要服务器上的CSI、checkpoint、ROS或百万行逐目标结果。', '',
                 '在公开文件夹根目录执行：', '', '```bash',
                 'python scripts/replay_figures.py --tables tables --output reproduced', '```', '',
                 '重画结果保存在reproduced/figures/，不会覆盖原图；figure_replay_manifest.json记录输入汇总表及两份脚本的哈希。该入口复现的是由已保存精确汇总值生成的图，不能在缺少逐目标数据时重新估计聚类置信区间或重新训练模型。']
        if self.notes:
            text += ['', '## 数据契约提醒', ''] + ['- '+s for s in sorted(set(self.notes))]
        (self.output/'RESULTS_CN.md').write_text('\n'.join(text)+'\n')
        manifest = {'report_source': str(Path(__file__).resolve()), 'report_sha256': file_hash(__file__),
                    'run_root': str(self.run_root.resolve()), 'sources': self.sources,
                    'missing_main_or_offgrid': missing, 'missing_stress': missing_stress, 'notes': sorted(set(self.notes)),
                    'bootstrap_repeats': self.repeats, 'bootstrap_unit_first_five': 'traj_id',
                    'bootstrap_unit_occlusion': 'channel_group (no trajectory fallback)',
                    'seed_aggregation': 'per-target linear NMSE mean before group/bootstrap',
                    'deep_fade_threshold_db': -10, 'figures': self.figures}
        (self.output/'report_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    def render_figures(self):
        """Only consume self.tables; also used by the portable CSV replay CLI."""
        configure_plotting()
        self.figures = []
        self.plot_main(); self.plot_offgrid(); self.plot_sampling(); self.plot_timestamps(); self.plot_staleness(); self.plot_occlusion()

    def run(self):
        self.load()
        if not self.groups:
            raise ValueError('No recognized raw per_target.csv files; no fabricated report will be emitted')
        self.build_tables()
        self.save_tables()
        self.render_figures()
        self.write_report()
        print('Report:', self.output/'RESULTS_CN.md')
        print('Sources:', len(self.sources), 'Figures:', len(self.figures), 'Main/off-grid missing:', len(self.completeness()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT/'runs/six_category_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    output = args.output or args.run_root/'report'
    Report(args.run_root, output, args.bootstrap_repeats, args.dpi).run()


if __name__ == '__main__':
    main()
