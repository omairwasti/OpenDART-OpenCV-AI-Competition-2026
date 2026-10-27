"""Prepare a human annotation gate before DRACO-specific perception development.

No predictions, perception configuration or metric labels are generated here.
The optional loopback server is read-only and exposes just HTML and sampled PNGs.
"""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import replay_draco_frozen as frozen

OUT=ROOT/'results/draco_specific_perception_2026-10-26'
SEED='OpenDART-DRACO-review-v1'
CLASSES=['TARGET_NOT_CONFIDENTLY_IDENTIFIABLE','DIMORPHOS_POINT_OR_SMALL_BODY',
         'DIMORPHOS_RESOLVED_BODY','PARTIAL_FRAME']
IMMUTABLE=['frame_index','mission_phase','product_stem','acquisition_utc','stratum',
           'split','is_partial','png_local_path','xml_local_path','png_sha256','xml_sha256']


def sample_positions(length,count):
    if count<1 or length<count: raise ValueError('Insufficient stratum coverage')
    return [round(i*(length-1)/(count-1)) for i in range(count)] if count>1 else [length//2]


def split_members(stems):
    ranked=sorted(stems,key=lambda s:hashlib.sha256((SEED+'|'+s).encode()).hexdigest())
    held=set(ranked[:round(len(ranked)/3)])
    return {stem:'held_out' if stem in held else 'development' for stem in stems}


def construct(index):
    impact=frozen.utc(frozen.IMPACT)
    bounds=[('terminal_2200_2224','2022-09-26T22:00:00Z','2022-09-26T22:24:00Z',10),
        ('terminal_2224_2245','2022-09-26T22:24:00Z','2022-09-26T22:45:00Z',10),
        ('terminal_2245_2300','2022-09-26T22:45:00Z','2022-09-26T23:00:00Z',8),
        ('terminal_2300_2310','2022-09-26T23:00:00Z','2022-09-26T23:10:00Z',8),
        ('early_final','2022-09-26T23:10:00Z',(impact-timedelta(seconds=60)).isoformat(),10),
        ('T60_T10',(impact-timedelta(seconds=60)).isoformat(),(impact-timedelta(seconds=10)).isoformat(),8),
        ('last10_complete_except_last',(impact-timedelta(seconds=10)).isoformat(),frozen.IMPACT,7)]
    selected=[]
    for name,a,b,count in bounds:
        eligible=[(i,r) for i,r in enumerate(index) if not r['is_partial']
            and r['product_stem']!=frozen.COMPLETE and frozen.utc(a)<=frozen.utc(r['acquisition_utc'])<frozen.utc(b)]
        samples=[eligible[i] for i in sample_positions(len(eligible),count)]
        splits=split_members([r['product_stem'] for _,r in samples])
        selected.extend((i,r,name,splits[r['product_stem']]) for i,r in samples)
    for stem,stratum in ((frozen.COMPLETE,'final_complete'),(frozen.PARTIAL,'final_partial')):
        i,r=next((i,r) for i,r in enumerate(index) if r['product_stem']==stem)
        selected.append((i,r,stratum,'analysis_only'))
    result=[]
    for i,r,stratum,split in sorted(selected,key=lambda item:item[0]):
        result.append(dict(frame_index=i,mission_phase=r['mission_phase'],product_stem=r['product_stem'],
            acquisition_utc=r['acquisition_utc'],stratum=stratum,split=split,is_partial=r['is_partial'],
            png_local_path=r['png_local_path'],xml_local_path=r['xml_local_path'],
            png_sha256=r['png_sha256'],xml_sha256=r['xml_sha256'],
            review_status='UNREVIEWED',label_class=None,target_center_uv=None,target_visible_box_xywh=None,
            didymos_center_uv=None,didymos_visible_box_xywh=None,confidence=None,note='',
            reviewer=None,reviewed_at_utc=None,human_inspection_attested=False))
    if len(result)!=63 or len({r['product_stem'] for r in result})!=63:
        raise ValueError('Unexpected/duplicate benchmark samples')
    return result


def schema():
    point={'anyOf':[{'type':'null'},{'type':'array','minItems':2,'maxItems':2,
        'items':{'type':'number','minimum':0,'maximum':1023}}]}
    box={'anyOf':[{'type':'null'},{'type':'array','minItems':4,'maxItems':4,
        'items':{'type':'number','minimum':0,'maximum':1024}}]}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Human DRACO review entry',
        'type':'object','required':IMMUTABLE+['review_status','label_class','target_center_uv',
            'target_visible_box_xywh','confidence','note','reviewer','reviewed_at_utc','human_inspection_attested'],
        'properties':{'review_status':{'enum':['UNREVIEWED','HUMAN_REVIEWED']},
            'label_class':{'enum':[None]+CLASSES},'is_partial':{'type':'boolean'},
            'target_center_uv':point,'didymos_center_uv':point,
            'target_visible_box_xywh':box,'didymos_visible_box_xywh':box,
            'confidence':{'enum':[None,'low','medium','high']},'note':{'type':'string'},
            'reviewer':{'type':['string','null']},'reviewed_at_utc':{'type':['string','null']},
            'human_inspection_attested':{'type':'boolean'}},
        '$comment':'Additional semantic rules are enforced by prepare_draco_review.validate_reviews; region boxes are visible extents, not full-body truth.'}


def geometry(value,box=False):
    if value is None:return
    if not isinstance(value,list) or len(value)!=(4 if box else 2) or any(
        isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value):
        raise ValueError('Invalid finite pixel geometry')
    if box:
        x,y,w,h=value
        if x<0 or y<0 or w<=0 or h<=0 or x+w>1024+1e-6 or y+h>1024+1e-6:raise ValueError('Visible box outside image')
    elif any(v<0 or v>1023 for v in value):raise ValueError('Point outside image')


def validate_reviews(rows,benchmark):
    expected={r['product_stem']:r for r in benchmark}
    if len({r['product_stem'] for r in rows})!=len(rows):raise ValueError('Duplicate annotations')
    for r in rows:
        if r['product_stem'] not in expected:raise ValueError('Not a benchmark sample')
        if any(r.get(k)!=expected[r['product_stem']][k] for k in IMMUTABLE):
            raise ValueError('Source metadata/split changed')
        for k in ('target_center_uv','didymos_center_uv'):geometry(r.get(k))
        for k in ('target_visible_box_xywh','didymos_visible_box_xywh'):geometry(r.get(k),True)
        status=r.get('review_status')
        if status=='UNREVIEWED':
            if r.get('human_inspection_attested') or r.get('reviewer') or r.get('reviewed_at_utc'):
                raise ValueError('Unreviewed entry cannot carry a review attestation')
            continue
        if status!='HUMAN_REVIEWED' or r.get('human_inspection_attested') is not True:
            raise ValueError('Explicit human review attestation required')
        if not isinstance(r.get('reviewer'),str) or not r['reviewer'].strip():raise ValueError('Reviewer required')
        frozen.utc(r.get('reviewed_at_utc') or '')
        if r.get('label_class') not in CLASSES or r.get('confidence') not in ('low','medium','high'):
            raise ValueError('Class and confidence required')
        category=r['label_class']
        if r['is_partial']!=(category=='PARTIAL_FRAME'):raise ValueError('Partial must be analyzed separately')
        if category=='TARGET_NOT_CONFIDENTLY_IDENTIFIABLE' and (r.get('target_center_uv') or r.get('target_visible_box_xywh')):
            raise ValueError('Ambiguous target cannot have target truth geometry')
        if category=='DIMORPHOS_POINT_OR_SMALL_BODY' and r.get('target_center_uv') is None:
            raise ValueError('Compact target requires defensible point')
        if category=='DIMORPHOS_RESOLVED_BODY' and r.get('target_visible_box_xywh') is None:
            raise ValueError('Resolved target requires visible region, not forced center')
        if not isinstance(r.get('note'),str) or not r['note'].strip():raise ValueError('Reviewer note required')
    return [r for r in rows if r['review_status']=='HUMAN_REVIEWED']


def build(out):
    index_path=ROOT/'data/draco/draco_terminal_to_impact_manifest.json'
    rows=construct(frozen.load_index(index_path))
    frozen.save(out/'annotation_schema.json',schema())
    frozen.save(out/'benchmark.json',rows)
    frozen.table(out/'benchmark.csv',rows)
    frozen.save(out/'reviewed_annotations.json',[])
    frozen.save(out/'split.json',{r['product_stem']:r['split'] for r in rows})
    policy=dict(status='AWAITING_HUMAN_REVIEW',method_implemented=False,configuration_frozen=False,
        full_replay_run=False,genuinely_human_reviewed_frames=0,metrics_computed=False,
        source_index_sha256=frozen.digest(index_path),seed=SEED,sampling='Evenly spaced integer rank within fixed UTC strata; no prediction/label-based sampling',
        split_counts=dict(Counter(r['split'] for r in rows)),stratum_counts=dict(Counter(r['stratum'] for r in rows)),
        split='Per-stratum SHA-256 ranking, round(n/3) held out; final complete/partial analysis only',
        review_protocol='Explicit original-image inspection, reviewer name, class/confidence/note and attestation; no automatic proposals',
        development_gate='Review development subset before implementing or freezing method; no held-out labels for tuning',
        held_out_gate='Import held-out labels only after development configuration/source hash is frozen; no tuning after evaluation',
        designation_gate='One click on a reviewed DEVELOPMENT compact-target frame only; exact UTC/point must be logged',
        scientific_boundary='Recorded image perception only; cannot validate causal guidance or flight readiness')
    frozen.save(out/'review_status.json',policy)
    display=[]
    import os
    from urllib.parse import quote
    for row in rows:
        relative=os.path.relpath(row['png_local_path'],out).replace('\\','/')
        display.append(dict(row,preview_url=quote(relative,safe='/')))
    template=Path(__file__).with_name('draco_annotation_review.html').read_text(encoding='utf-8')
    data=json.dumps(display,allow_nan=False).replace('<','\\u003c')
    path=out/'review.html'
    if path.exists():raise FileExistsError(path)
    path.write_text(template.replace('__BENCHMARK_DATA__',data),encoding='utf-8')
    print(json.dumps(policy,indent=2))


def serve(out,port):
    rows=json.loads((out/'benchmark.json').read_text())
    allowed={'/images/'+r['product_stem']+'.png':Path(r['png_local_path']) for r in rows}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=unquote(urlsplit(self.path).path)
            if route in ('/','/review.html'):
                file=out/'review.html';mime='text/html; charset=utf-8'
            elif route in allowed:
                file=allowed[route];mime='image/png'
            else:self.send_error(404);return
            data=file.read_bytes();self.send_response(200)
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Read-only review server: http://127.0.0.1:{port}/review.html',flush=True)
    server.serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['build','serve','validate','verify'],required=True)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--annotations',type=Path)
    args=parser.parse_args()
    if args.phase=='build':build(args.out)
    elif args.phase=='serve':serve(args.out,args.port)
    elif args.phase=='verify':frozen.verify(args.out)
    else:
        if args.annotations is None:parser.error('--annotations required')
        reviewed=validate_reviews(json.loads(args.annotations.read_text(encoding='utf-8')),
                                  json.loads((args.out/'benchmark.json').read_text()))
        print(json.dumps(dict(attested_human_reviewed_frames=len(reviewed),split_counts=dict(Counter(r['split'] for r in reviewed)),
            note='Attestation validation, not independent proof of review or annotation correctness; no metrics computed.'),indent=2))
