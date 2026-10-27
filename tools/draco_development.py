"""Development-only DRACO evaluation, freeze and qualitative replay harness.

No held-out annotation file is discovered, opened or accepted. The only label
input is the validated development export; online tracking receives one explicit
designation record at most. Frozen baseline outputs are read without rerunning.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import draco_perception as method
import replay_draco_frozen as io

OUT=io.ROOT/'results/draco_specific_perception_2026-10-26/development_01'
CONFIG=Path(__file__).with_name('draco_perception_config_v1.json')
LABEL='DEVELOPMENT SET - NOT HELD-OUT EVALUATION'


def labels(out):
    rows=json.loads((out/'validated_development_annotations.json').read_text())
    if not rows or any(r['split']!='development' or r['review_status']!='HUMAN_REVIEWED' for r in rows):
        raise ValueError('Reviewed development labels only')
    return rows


def designation(annotations):
    eligible=[r for r in annotations if r['confidence']=='high' and r['target_visible_box_xywh'] is not None
              and 15<=max(r['target_visible_box_xywh'][2:])<128]
    if not eligible:raise ValueError('No clearly reviewed one-time designation frame')
    row=min(eligible,key=lambda r:r['frame_index']);box=row['target_visible_box_xywh']
    return dict(frame_index=row['frame_index'],product_stem=row['product_stem'],acquisition_utc=row['acquisition_utc'],
        visible_box_xywh=box,center_uv=[box[0]+box[2]/2,box[1]+box[3]/2],
        source='One explicitly reviewed DEVELOPMENT region; earliest high-confidence compact visible region >=15px',
        initialization='Choose current image component whose centroid lies inside this one box, nearest box center',
        manual_initializations=1,later_manual_corrections=0)


def iou(a,b):
    if a is None or b is None:return 0.
    x=max(a[0],b[0]);y=max(a[1],b[1]);right=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    intersection=max(0.,right-x)*max(0.,bottom-y)
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection) if intersection else 0.


def evaluate(annotations,predictions,config,exclude_designation=None):
    records=[]
    for a in annotations:
        p=predictions[a['frame_index']];point=a['target_center_uv'];box=a['target_visible_box_xywh']
        eligible=exclude_designation is None or a['frame_index']>exclude_designation
        value=dict(frame_index=a['frame_index'],product_stem=a['product_stem'],class_name=a['label_class'],
            confidence=a['confidence'],eligible=eligible,selected=p['selection_available'],state=p['state'],
            centroid_error_px=None,point_within_tolerance=None,region_iou=None,region_pass=None)
        if eligible and a['label_class']=='DIMORPHOS_POINT_OR_SMALL_BODY':
            if p['target_center_uv'] is not None:
                value['centroid_error_px']=float(np.linalg.norm(np.asarray(p['target_center_uv'])-point))
                value['point_within_tolerance']=value['centroid_error_px']<=config['point_evaluation_tolerance_px']
            else:value['point_within_tolerance']=False
        if eligible and a['label_class']=='DIMORPHOS_RESOLVED_BODY':
            value['region_iou']=iou(p['visible_region_xywh'],box)
            value['region_pass']=value['region_iou']>=config['region_evaluation_iou_threshold']
        records.append(value)
    compact=[r for r in records if r['eligible'] and r['class_name']=='DIMORPHOS_POINT_OR_SMALL_BODY']
    regions=[r for r in records if r['eligible'] and r['class_name']=='DIMORPHOS_RESOLVED_BODY']
    errors=[r['centroid_error_px'] for r in compact if r['centroid_error_px'] is not None]
    uncertain=[r for r in records if r['eligible'] and r['class_name']=='TARGET_NOT_CONFIDENTLY_IDENTIFIABLE']
    summary=dict(label=LABEL,reviewed_samples=len(annotations),excluded_pre_or_at_designation=sum(not r['eligible'] for r in records),
        compact=dict(n=len(compact),selected_n=sum(r['selected'] for r in compact),
            within_5px_n=sum(r['point_within_tolerance'] is True for r in compact),
            selected_outside_5px_n=sum(r['point_within_tolerance'] is False and r['selected'] for r in compact),
            unavailable_n=sum(not r['selected'] for r in compact),
            centroid_error_n=len(errors),centroid_error_median_px=float(np.median(errors)) if errors else None,
            centroid_errors_px=errors,confidence_counts=dict(Counter(r['confidence'] for r in compact))),
        regions=dict(n=len(regions),selected_n=sum(r['selected'] for r in regions),
            median_iou=float(np.median([r['region_iou'] for r in regions])) if regions else None,
            iou_ge_0_3_n=sum(r['region_pass'] for r in regions),region_ious=[r['region_iou'] for r in regions]),
        uncertain=dict(n=len(uncertain),selected_n=sum(r['selected'] for r in uncertain),
            note='Selections on ambiguous labels are not counted as semantic false positives or correct identification'),
        point_tolerance_px=config['point_evaluation_tolerance_px'],region_iou_threshold=config['region_evaluation_iou_threshold'],
        not_reported=['precision','recall','mAP','generalization accuracy'],designation_frame_excluded=exclude_designation)
    return records,summary


def baseline(out):
    annotations=labels(out);cfg=method.load_config();raw=json.loads((io.ROOT/'results/draco_frozen_replay_2026-10-26/frames.json').read_text())
    predictions={}
    for row in raw:
        i=row['selected_candidate_index'];c=None if i is None else row['candidates'][i]
        predictions[row['frame_index']]=dict(selection_available=c is not None,state=row['track_status'],
            target_center_uv=None if c is None else c['centroid_px'],visible_region_xywh=None if c is None else c['bbox_xywh'])
    records,summary=evaluate(annotations,predictions,cfg)
    io.save(out/'frozen_baseline_development_metrics.json',summary);io.table(out/'frozen_baseline_development_samples.csv',records)
    _,matched=evaluate(annotations,predictions,cfg,designation(annotations)['frame_index'])
    io.save(out/'frozen_baseline_post_designation_metrics.json',matched)
    print(json.dumps(summary,indent=2))


def run(out,full=False):
    if full:
        freeze=json.loads((out/'FREEZE.json').read_text())
        if io.digest(out/'frozen_configuration.json')!=freeze['configuration_sha256']:
            raise RuntimeError('Frozen configuration artifact mismatch')
        for name,sha in freeze['source_hashes'].items():
            if io.digest(io.ROOT/name)!=sha:raise RuntimeError('Frozen source/config mismatch')
        cfg=json.loads((out/'frozen_configuration.json').read_text())['parameters']
        click=json.loads((out/'frozen_configuration.json').read_text())['designation']
    else:
        cfg=method.load_config();click=designation(labels(out))
        io.save(out/'development_configuration_v1.json',dict(parameters=cfg,designation=click))
    index=io.load_index(io.ROOT/'data/draco/draco_terminal_to_impact_manifest.json')
    last=len(index)-1 if full else max(r['frame_index'] for r in labels(out))
    feature=method.Features(cfg);tracks={mode:method.Track(cfg,mode,click if mode=='human_designated' else None)
        for mode in ('autonomous','human_designated')}
    cv=method.require_opencv();results=[];begin=time.perf_counter()
    for i,item in enumerate(index[:last+1]):
        image=cv.imread(item['png_local_path'],cv.IMREAD_UNCHANGED)
        elapsed=(io.utc(item['acquisition_utc'])-io.utc(index[0]['acquisition_utc'])).total_seconds()
        obs=feature.update(image,elapsed)
        row=dict(frame_index=i,product_stem=item['product_stem'],mission_phase=item['mission_phase'],
            acquisition_utc=item['acquisition_utc'],elapsed_s=elapsed,
            relative_to_impact_s=(io.utc(item['acquisition_utc'])-io.utc(io.IMPACT)).total_seconds(),
            is_partial=item['is_partial'],png_local_path=item['png_local_path'],
            raw_candidate_count=obs['raw_candidate_count'],kept_candidate_count=len(obs['candidates']),
            persistent_rejected_count=len(obs['rejected_persistent']),image_shift_px=obs['image_shift_px'],
            phase_response=obs['phase_response'],feature_wall_s=obs['feature_wall_s'],
            **{mode:track.update(obs,elapsed,i,item['is_partial']) for mode,track in tracks.items()})
        results.append(row)
        if i%200==0 or i==last:print(f"{'frozen replay' if full else 'development causal prefix'} {i+1}/{last+1}: {row['autonomous']['state']} / {row['human_designated']['state']}",flush=True)
    prefix='qualitative_frozen' if full else 'development_v1'
    io.save(out/(prefix+'_frames.json'),results)
    io.table(out/(prefix+'_frames.csv'),[{k:(v if not isinstance(v,dict) else json.dumps(v)) for k,v in r.items()} for r in results])
    summary=dict(label='FROZEN QUALITATIVE REPLAY - NO HELD-OUT TRUTH' if full else LABEL,
        frames=len(results),total_wall_s=time.perf_counter()-begin,designation=click,
        feature_ms_mean=float(np.mean([r['feature_wall_s']*1000 for r in results])),
        persistent_rejected_components=sum(r['persistent_rejected_count'] for r in results),
        modes={mode:dict(states=dict(Counter(r[mode]['state'] for r in results)),
            events=dict(Counter(e for r in results for e in r[mode]['events'])),
            available_nonpartial=sum(r[mode]['selection_available'] for r in results if not r['is_partial']),
            resolved_multicomponent_frames=sum(r[mode]['resolved_body'] and r[mode]['region_component_count']>1 for r in results),
            final=results[-1][mode]) for mode in tracks})
    io.save(out/(prefix+'_summary.json'),summary)
    if not full:
        annotations=labels(out);comparison={}
        for mode in tracks:
            predictions={r['frame_index']:r[mode] for r in results}
            records,metrics=evaluate(annotations,predictions,cfg,click['frame_index'] if mode=='human_designated' else None)
            io.save(out/(mode+'_development_metrics.json'),metrics)
            io.table(out/(mode+'_development_samples.csv'),records);comparison[mode]=metrics
            if mode=='autonomous':
                _,matched=evaluate(annotations,predictions,cfg,click['frame_index'])
                io.save(out/'autonomous_post_designation_metrics.json',matched)
        io.save(out/'development_comparison.json',comparison)
        print(json.dumps(comparison,indent=2))


def freeze(out):
    cfg=json.loads((out/'development_configuration_v1.json').read_text())
    comparison=json.loads((out/'development_comparison.json').read_text())
    baseline_metrics=json.loads((out/'frozen_baseline_development_metrics.json').read_text())
    if comparison['autonomous']['compact']['within_5px_n']<=baseline_metrics['compact']['within_5px_n']:
        raise RuntimeError('No meaningful development point improvement; stop rather than forcing freeze')
    if comparison['autonomous']['regions']['median_iou']<=baseline_metrics['regions']['median_iou']:
        raise RuntimeError('No development region improvement; stop rather than forcing freeze')
    source=[Path(__file__),Path(method.__file__),CONFIG,io.ROOT/'opendart/vision.py',Path(io.__file__)]
    source_hashes={str(p.resolve().relative_to(io.ROOT)):io.digest(p) for p in source}
    frozen_cfg=dict(**cfg,label=LABEL,source_hashes=source_hashes,
        development_annotation_sha256=io.digest(out/'validated_development_annotations.json'),
        opencv=method.require_opencv().__version__,numpy=np.__version__,python=sys.version,
        state_rules=dict(search='Three consecutive unique separated secondary hypotheses, >=30px2 and primary >=2.5x area',
            tracking='Past/current phase-correlation translation and position/area continuity',
            lost='No accepted observation; no selected target emitted',
            reacquiring='After five missed frames discard stale gate; require fresh three-frame two-body hypothesis; identity link downgraded',
            resolved='Extent >=64px for two consecutive detections, then collective foreground envelope; no single-component requirement',
            partial='Curated metadata flag; transmitted region only, no complete-frame metrics'),
        held_out_truth_consumed=False,no_tuning_after_freeze=True,
        boundary='Recorded image perception, never causal closed-loop guidance')
    io.save(out/'frozen_configuration.json',frozen_cfg)
    io.save(out/'FREEZE.json',dict(frozen_at_utc=io.now() if hasattr(io,'now') else __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        source_hashes=source_hashes,configuration_sha256=io.digest(out/'frozen_configuration.json'),
        held_out_truth_consumed=False,required_stop='Request separate human-reviewed held-out JSON; no evaluation until supplied'))
    directory=out/'frozen_source';directory.mkdir(exist_ok=False)
    for p in source:shutil.copy2(p,directory/p.name)
    print(json.dumps(json.loads((out/'FREEZE.json').read_text()),indent=2))


def render(out):
    cv=method.require_opencv()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((out/'qualitative_frozen_frames.json').read_text())
    annotations=labels(out);lookup={r['frame_index']:r for r in rows}
    def panel(row,mode,width=480,annotation=None):
        image=cv.imread(row['png_local_path'],cv.IMREAD_UNCHANGED);canvas=cv.cvtColor(image,cv.COLOR_GRAY2BGR);p=row[mode]
        if p['visible_region_xywh'] is not None:
            x,y,w,h=map(lambda v:int(round(v)),p['visible_region_xywh']);cv.rectangle(canvas,(x,y),(x+w,y+h),(50,220,255),2)
        if annotation is not None:
            if annotation['target_center_uv'] is not None:cv.drawMarker(canvas,tuple(np.rint(annotation['target_center_uv']).astype(int)),(90,255,90),cv.MARKER_CROSS,18,2)
            elif annotation['target_visible_box_xywh'] is not None:
                x,y,w,h=map(lambda v:int(round(v)),annotation['target_visible_box_xywh']);cv.rectangle(canvas,(x,y),(x+w,y+h),(90,255,90),2)
        canvas=cv.resize(canvas,(width,width),interpolation=cv.INTER_AREA)
        result=np.zeros((width+132,width,3),np.uint8);result[:width]=canvas
        lines=['RECORDED DRACO REPLAY - NOT CLOSED LOOP',mode+' | '+p['state'],
            row['acquisition_utc']+f" T{row['relative_to_impact_s']:.3f}s",p['identity_status'],
            'PARTIAL: transmitted region only' if row['is_partial'] else LABEL if annotation else
            'REACQUIRED' if 'REACQUIRED' in p['events'] else 'HUMAN DESIGNATION' if 'HUMAN_DESIGNATION' in p['events'] else p['representation'] or 'No selected region']
        for j,line in enumerate(lines):cv.putText(result,line,(7,width+20+j*22),cv.FONT_HERSHEY_SIMPLEX,.36,(70,150,255) if row['is_partial'] else (230,230,230),1,cv.LINE_AA)
        return result
    chosen=[r for r in annotations if r['frame_index'] in (2042,2534,3110,3456,3764,3825,3906,3947,3984,3991,3992,3994)]
    directory=out/'representatives';directory.mkdir(exist_ok=False)
    for a in chosen:
        image=panel(lookup[a['frame_index']],'autonomous',640,a)
        cv.imwrite(str(directory/f"development_{a['frame_index']:04d}.png"),image)
    panels=[panel(lookup[a['frame_index']],'autonomous',360,a) for a in chosen]
    cv.imwrite(str(out/'development_contact_sheet.png'),np.vstack([np.hstack(panels[i:i+3]) for i in range(0,len(panels),3)]))
    for i in (3998,3999):cv.imwrite(str(directory/f'qualitative_{i}.png'),np.hstack([panel(rows[i],m) for m in ('autonomous','human_designated')]))
    fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    t=[r['relative_to_impact_s']/60 for r in rows]
    for mode in ('autonomous','human_designated'):
        axes[0].plot(t,[int(r[mode]['selection_available']) for r in rows],lw=.6,label=mode)
        axes[1].plot(t,[max(r[mode]['visible_region_xywh'][2:]) if r[mode]['visible_region_xywh'] else np.nan for r in rows],lw=.7,label=mode)
    axes[2].plot(t,[r['persistent_rejected_count'] for r in rows],lw=.7,label='Past-only rejected small components')
    axes[0].set_ylabel('Selected region available');axes[1].set_ylabel('Visible extent (px)');axes[2].set_ylabel('Rejected components')
    for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set_xlabel('Minutes relative to approximate impact')
    fig.suptitle('FROZEN QUALITATIVE REPLAY - NO HELD-OUT TRUTH\nRecorded DRACO imagery, not closed loop')
    fig.savefig(out/'qualitative_diagnostics.png',dpi=130);plt.close(fig)
    comparison=json.loads((out/'development_comparison.json').read_text())
    base=json.loads((out/'frozen_baseline_development_metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].bar(['Frozen baseline','DRACO autonomous'],[base['compact']['within_5px_n'],comparison['autonomous']['compact']['within_5px_n']])
    axes[0].set(ylabel='Reviewed point matches within 5px (n=9)',ylim=(0,9))
    for name,color in [('frozen_baseline','gray'),('autonomous','tab:blue'),('human_designated','tab:orange')]:
        metric=base if name=='frozen_baseline' else comparison[name]
        values=metric['regions']['region_ious']
        axes[1].scatter(range(len(values)),values,label=f"{name} (n={len(values)})",s=20,color=color)
    axes[1].set(xlabel='Sample index within each eligible development subset',ylabel='Visible-region IoU',ylim=(0,1));axes[1].legend(fontsize=8)
    fig.suptitle(LABEL+'\nHuman designation initialization and earlier frames excluded from its metrics')
    fig.savefig(out/'development_metrics.png',dpi=140);plt.close(fig)
    path=out/'draco_specific_recorded_replay.mp4'
    if path.exists():raise FileExistsError(path)
    writer=cv.VideoWriter(str(path),cv.VideoWriter_fourcc(*'mp4v'),60.,(960,612))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    try:
        for i,row in enumerate(rows):
            writer.write(np.hstack([panel(row,m) for m in ('autonomous','human_designated')]))
            if i%500==0:print('render',i,flush=True)
    finally:writer.release()
    capture=cv.VideoCapture(str(path));decoded=0
    while True:
        ok,frame=capture.read()
        if not ok:break
        if frame.shape[:2]!=(612,960):raise RuntimeError('Unexpected video structure')
        decoded+=1
    capture.release()
    if decoded!=4000:raise RuntimeError('Incomplete review video')
    io.save(out/'render_verification.json',dict(decoded_frames=decoded,fps=60,playback_s=4000/60,
        bytes=path.stat().st_size,sha256=io.digest(path),source_intensities_preserved_for_display=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['baseline','develop','freeze','replay','render','verify'],required=True)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    {'baseline':baseline,'develop':lambda p:run(p,False),'freeze':freeze,'replay':lambda p:run(p,True),
     'render':render,'verify':io.verify}[args.phase](args.out)
