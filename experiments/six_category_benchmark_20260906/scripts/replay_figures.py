"""Portable redraw using summary tables only, without CSI, weights, or raw rows.

Keep this file next to report.py. With the published folder layout:
  python scripts/replay_figures.py --tables tables --output reproduced
Dependencies: numpy, pandas, matplotlib. No PyTorch, ROS, AirSim, or Sionna.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

try:
    from .report import Report, file_hash
except ImportError:
    from report import Report, file_hash


PLOT_TABLES = ['overall', 'by_horizon', 'offgrid_horizon_bands', 'occlusion_groups',
               'timestamp_paired_changes', 'sampling_and_staleness_paired_changes']


def replay_figures(tables_dir, output, dpi=600):
    tables_dir = Path(tables_dir)
    paths = sorted(tables_dir.glob('*.csv'))
    if not paths:
        raise ValueError('No summary CSVs found in '+str(tables_dir))
    if not (tables_dir/'overall.csv').exists():
        raise ValueError('overall.csv is required; do not pass the original per-target result folder')
    report = Report(tables_dir.parent, output, dpi=dpi)
    # No call to Report.load(), build_tables(), or any prediction/evaluation code.
    report.tables = {name: pd.DataFrame() for name in PLOT_TABLES}
    sources = []
    for path in paths:
        table = pd.read_csv(path)
        for key in ['seeds_complete', 'all_seeds_complete', 'complete_six_families']:
            if key in table:
                table[key] = table[key].fillna(False).astype(str).str.lower().isin(['true', '1', '1.0'])
        report.tables[path.stem] = table
        sources.append({'filename': path.name, 'sha256': file_hash(path), 'rows': len(table)})
    if not report.tables['overall'].empty and report.tables['by_horizon'].empty:
        # plot_main needs horizon rows if any complete main results are present.
        overall = report.tables['overall']
        if ((overall['experiment']=='main') & overall['seeds_complete']).any():
            raise ValueError('by_horizon.csv is required to redraw complete main results')
    report.render_figures()
    manifest = {'input_tables': sources, 'figures': report.figures, 'dpi': dpi,
                'summary_only': True, 'raw_per_target_read': False, 'weights_or_csi_read': False,
                'numpy_version': np.__version__, 'pandas_version': pd.__version__,
                'matplotlib_version': matplotlib.__version__,
                'report_script_sha256': file_hash(Path(__file__).with_name('report.py')),
                'replay_script_sha256': file_hash(__file__)}
    (Path(output)/'figure_replay_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Redrawn', len(report.figures), 'figure groups from summary CSVs:', Path(output)/'figures')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True, help='Published report/tables directory')
    parser.add_argument('--output', type=Path, help='Defaults to tables-parent/reproduced; original images remain untouched')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    replay_figures(args.tables, args.output or args.tables.parent/'reproduced', args.dpi)


if __name__ == '__main__':
    main()
