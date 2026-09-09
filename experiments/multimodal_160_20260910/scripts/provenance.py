import json,platform,sys,importlib.metadata,xml.etree.ElementTree as ET
from pathlib import Path
from prepare import ROOT,PROJECT,save,digest
def main():
    phy=json.loads((ROOT/'physical_config.json').read_text());scene=Path(phy['scene_path']);files=[scene]
    for node in ET.parse(scene).iter('string'):
        if node.attrib.get('name')=='filename':
            p=Path(node.attrib['value']);files.append(p if p.is_absolute() else scene.parent/p)
    code=[PROJECT/'union_complete_candidates.py',Path('/root/autodl-tmp/csi-irregular-benchmark/rt_runtime/snapshots_geometry.py')]
    versions={}
    for package in ['numpy','torch','sionna-rt','mitsuba','drjit','scipy']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not found'
    save(ROOT/'provenance.json',dict(python=sys.version,platform=platform.platform(),versions=versions,radio_assets_sha256={str(p):digest(p) for p in files},external_code_sha256={str(p):digest(p) for p in code},
        trajectory_source_config_sha256=digest(PROJECT/'trajectory_160_preview_v3/config.json'),
        selected_indices_original_grid=[0,8,17,25,34,42,51,59,68,76,85,93,102,110,119,127],
        note='Predictor source hashes and frozen CNN weight hashes saved separately in run/feature metadata.'))
if __name__=='__main__':main()
