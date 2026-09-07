"""Regenerate artifact checksums, excluding local Python caches and the manifest itself."""
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parents[1]
files={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='ARTIFACT_MANIFEST.json':continue
    files[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rows=list(csv.DictReader((ROOT/'tables/overall.csv').open()))
manifest={'run':'first_five_affine_s17_plus_occlusion_v4_affine_s17','seed':17,'trained_models':36,'evaluation_combinations':len(rows),'figure_groups':11,'occlusion_version':'v4_nearby','missing_v4_cases':'N/E; old occlusion rows not retained','files':files}
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
