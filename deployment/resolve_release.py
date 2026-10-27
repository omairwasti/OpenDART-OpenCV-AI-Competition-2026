"""Resolve portable audited release metadata; does not contact AWS."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
r=json.loads((root/'release.json').read_text());r['archive']=str((root/Path(r['archive']).name).resolve())
assert Path(r['archive']).is_file()
if a.out.exists():raise FileExistsError(a.out)
if a.out.resolve().parent!=root:raise ValueError('Write local metadata beside release_manifest.json')
a.out.write_text(json.dumps(r,indent=2)+'\n');print('Local metadata prepared; no AWS action performed')
