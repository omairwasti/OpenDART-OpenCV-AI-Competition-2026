"""Frozen, image-only DRACO baseline. No scientific code/configuration changes.

Run detect, analyze, render, verify in order into a fresh evidence directory.
Detection calls the existing AsteroidDetector; association calls AsteroidTracker
without external priors. Original full-resolution pixels enter detection.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opendart.vision import AsteroidDetector, AsteroidTracker, Candidate, require_opencv
import opendart.vision as vision

IMPACT = '2022-09-26T23:14:24.183Z'
PARTIAL = 'dart_0401930050_41838_02_iof'
COMPLETE = 'dart_0401930049_43695_02_iof'
WARNING = 'RECORDED DRACO REPLAY - NOT CLOSED LOOP'
DEFAULT_OUT = ROOT/'results/draco_frozen_replay_2026-10-26'


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('UTC requires timezone')
    return parsed.astimezone(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing evidence: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def table(path, rows):
    if Path(path).exists():
        raise FileExistsError(path)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, allow_nan=False) if isinstance(v,(list,dict)) else v
                             for k,v in row.items()})


def load_index(path, expected_count=4000):
    rows = json.loads(Path(path).read_text())
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError('Unexpected source index count')
    if not rows or len({r['product_stem'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate source products')
    if any(utc(a['acquisition_utc']) >= utc(b['acquisition_utc']) for a,b in zip(rows,rows[1:])):
        raise ValueError('Index must already be strictly ordered by XML UTC')
    for row in rows:
        if bool(row['is_partial']) != (row['product_stem'] == PARTIAL):
            raise ValueError('Curated partial status must be preserved')
    return rows


def frozen_config():
    return dict(detector_defaults={k:p.default for k,p in inspect.signature(AsteroidDetector).parameters.items()},
                tracker_defaults={k:p.default for k,p in inspect.signature(AsteroidTracker).parameters.items()},
                vision_path=str(Path(vision.__file__).resolve()),vision_sha256=digest(vision.__file__),
                adapter_sha256=digest(__file__),python=platform.python_version(),
                opencv=require_opencv().__version__,numpy=np.__version__,
                detection_input='Original 1024x1024 uint8 grayscale PNG; no adapter preprocessing or validity mask',
                frozen_internal_preprocessing='normalize, Gaussian blur, threshold, morphological opening: unchanged vision.py',
                association='Default image-only AsteroidTracker, no external predicted_uv/covariance; no EKF/guidance',
                coordinate_convention='Stored PNG zero-based u right, v down; PDS Bottom to Top noted, no transform applied',
                identity='Selected candidate/track only; no semantic identity or ground truth',
                impact_reference_utc=IMPACT,scientific_boundary=WARNING)


def decode_candidate(data):
    return Candidate(np.asarray(data['centroid_px']),data['radius_px'],
        np.asarray(data['covariance_px2']),data['area_px2'],tuple(data['bbox_xywh']),
        data['score'],data['touches_border'],
        None if data['ellipse_axes_px'] is None else tuple(data['ellipse_axes_px']),data['ellipse_angle_deg'])


def detect(index, out):
    cv = require_opencv()
    if int(cv.__version__.split('.')[0]) != 5:
        raise RuntimeError('OpenCV 5 required')
    if (out/'detector_frames.json').exists():
        raise FileExistsError('Detector evidence already exists')
    source = load_index(index)
    config = frozen_config()
    config.update(index_path=str(index.resolve()), index_sha256=digest(index))
    save(out/'configuration.json', config)
    detector = AsteroidDetector()
    result = []
    start = time.perf_counter()
    for i,item in enumerate(source):
        before_io = time.perf_counter()
        image = cv.imread(item['png_local_path'],cv.IMREAD_UNCHANGED)
        if image is None or image.shape != (1024,1024) or image.dtype != np.uint8:
            raise ValueError('Unexpected native input: '+item['product_stem'])
        begin = time.perf_counter()
        candidates = detector.detect(image)
        elapsed = time.perf_counter()-begin
        row = dict(frame_index=i,mission_phase=item['mission_phase'],product_stem=item['product_stem'],
            acquisition_utc=item['acquisition_utc'],
            elapsed_s=(utc(item['acquisition_utc'])-utc(source[0]['acquisition_utc'])).total_seconds(),
            relative_to_impact_s=(utc(item['acquisition_utc'])-utc(IMPACT)).total_seconds(),
            is_partial=item['is_partial'],pds_bad_image=item['pds_bad_image'],
            image_completeness=item['image_completeness'],
            candidate_count=len(candidates),candidates=[c.to_dict() for c in candidates],
            detector_status='CANDIDATES' if candidates else 'NO_CANDIDATES',failure_reason=None,
            detector_wall_s=elapsed,read_wall_s=begin-before_io,
            png_local_path=item['png_local_path'],xml_local_path=item['xml_local_path'])
        result.append(row)
        if i%200 == 0 or i == len(source)-1:
            print(f'detect {i+1}/{len(source)}: {len(candidates)} candidates; {elapsed:.4f}s',flush=True)
        if elapsed > 5:
            raise RuntimeError('Unexpectedly excessive detector runtime; stopping without tuning')
    save(out/'detector_frames.json',result)
    table(out/'detector_frames.csv',[{k:v for k,v in row.items() if k!='candidates'} for row in result])
    table(out/'detector_candidates.csv',[dict(frame_index=r['frame_index'],product_stem=r['product_stem'],
        acquisition_utc=r['acquisition_utc'],candidate_index=i,**c) for r in result for i,c in enumerate(r['candidates'])])
    save(out/'detector_run.json',dict(frames=len(result),total_wall_s=time.perf_counter()-start,
        complete=True,detector_errors=0))


def associate(rows):
    tracker = AsteroidTracker()
    previous_status = 'SEARCHING'
    previous_selected = None
    previous_index = None
    events = []
    for row in rows:
        candidates = [decode_candidate(c) for c in row['candidates']]
        started = time.perf_counter()
        selected = tracker.update(candidates,row['elapsed_s'])
        row['association_wall_s'] = time.perf_counter()-started
        index = next((i for i,c in enumerate(candidates) if c is selected),None)
        row.update(selected_candidate_index=index,track_status=tracker.status,
                   missed_frames=tracker.missed_frames,
                   track_prediction_uv_px=None if tracker.x is None else tracker.x[:2].tolist(),
                   innovation_chi2=float(tracker.last_innovation_chi2) if selected is not None and
                       np.isfinite(tracker.last_innovation_chi2) else None,
                   association_failure_reason=None if selected is not None else
                       'NO_CANDIDATES' if not candidates else
                       'NO_IMAGE_ONLY_ACQUISITION_HYPOTHESIS' if tracker.x is None else
                       'NO_CANDIDATE_PASSED_FROZEN_POSITION_AND_SIZE_GATES',
                   selected_centroid_px=None if selected is None else selected.centroid_px.tolist(),
                   selected_radius_px=None if selected is None else selected.radius_px,
                   selected_area_px2=None if selected is None else selected.area_px2)
        kinds=[]
        if tracker.status in ('ACQUIRED','REACQUIRED'): kinds.append(tracker.status)
        if tracker.status=='LOST' and previous_status!='LOST': kinds.append('ENTERED_LOST')
        if tracker.status=='COAST' and previous_status not in ('COAST','LOST'): kinds.append('ENTERED_COAST')
        # Per-frame area rank is not persistent physical identity. Report it as
        # a rank change, never as a measured body-identity switch.
        if index is not None and previous_selected is not None and index!=previous_index:
            kinds.append('SELECTED_AREA_RANK_CHANGED_NOT_IDENTITY_SWITCH')
        row['events']=kinds
        for kind in kinds:
            events.append(dict(frame_index=row['frame_index'],acquisition_utc=row['acquisition_utc'],
                product_stem=row['product_stem'],event=kind,selected_candidate_index=index))
        previous_status=tracker.status
        previous_selected=selected
        previous_index=index
    return events


def runs(rows, predicate):
    result=[]; current=[]
    for row in rows:
        if predicate(row): current.append(row)
        elif current:
            result.append(current); current=[]
    if current: result.append(current)
    return [dict(first_frame=r[0]['frame_index'],last_frame=r[-1]['frame_index'],frames=len(r),
        first_utc=r[0]['acquisition_utc'],last_utc=r[-1]['acquisition_utc'],
        sampled_span_s=r[-1]['elapsed_s']-r[0]['elapsed_s']) for r in result]


def statistics(rows):
    selected=[r for r in rows if r['selected_candidate_index'] is not None]
    ms=np.array([r['detector_wall_s']*1000 for r in rows])
    absent=runs(rows,lambda r:r['candidate_count']==0)
    tracks=runs(rows,lambda r:r['selected_candidate_index'] is not None)
    unselected=runs(rows,lambda r:r['selected_candidate_index'] is None)
    return dict(frames=len(rows),frames_with_candidates=sum(r['candidate_count']>0 for r in rows),
        no_candidate_frames=sum(r['candidate_count']==0 for r in rows),
        candidate_frame_fraction=sum(r['candidate_count']>0 for r in rows)/len(rows) if rows else None,
        candidate_count_distribution=dict(sorted(Counter(r['candidate_count'] for r in rows).items())),
        selected_frames=len(selected),track_status_counts=dict(Counter(r['track_status'] for r in rows)),
        event_counts=dict(Counter(e for r in rows for e in r['events'])),
        no_candidate_runs=absent,longest_no_candidate_run=max(absent,key=lambda r:r['sampled_span_s'],default=None),
        selected_runs=tracks,longest_selected_run=max(tracks,key=lambda r:r['sampled_span_s'],default=None),
        longest_unselected_run=max(unselected,key=lambda r:r['sampled_span_s'],default=None),
        detector_ms=dict(mean=float(ms.mean()),median=float(np.median(ms)),p95=float(np.percentile(ms,95)),
            min=float(ms.min()),max=float(ms.max()),throughput_fps=1000/float(ms.mean())) if len(ms) else None,
        largest_candidate_radius_px_range=[min(r['candidates'][0]['radius_px'] for r in rows if r['candidates']),
            max(r['candidates'][0]['radius_px'] for r in rows if r['candidates'])] if any(r['candidates'] for r in rows) else None,
        selected_radius_px_range=[min(r['selected_radius_px'] for r in selected),max(r['selected_radius_px'] for r in selected)] if selected else None)


def windows(rows):
    impact=utc(IMPACT)
    from datetime import timedelta
    bounds=[('22:00-22:24','2022-09-26T22:00:00Z','2022-09-26T22:24:00Z'),
        ('22:24-22:45','2022-09-26T22:24:00Z','2022-09-26T22:45:00Z'),
        ('22:45-23:00','2022-09-26T22:45:00Z','2022-09-26T23:00:00Z'),
        ('23:00-23:10','2022-09-26T23:00:00Z','2022-09-26T23:10:00Z'),
        ('FINAL to T-60','2022-09-26T23:10:00Z',(impact-timedelta(seconds=60)).isoformat()),
        ('T-60 to T-10',(impact-timedelta(seconds=60)).isoformat(),(impact-timedelta(seconds=10)).isoformat()),
        ('Final 10 seconds',(impact-timedelta(seconds=10)).isoformat(),IMPACT)]
    return {name:statistics([r for r in rows if not r['is_partial'] and utc(a)<=utc(r['acquisition_utc'])<utc(b)])
            for name,a,b in bounds}


def analyze(out):
    config=json.loads((out/'configuration.json').read_text())
    if config['vision_sha256']!=digest(vision.__file__): raise RuntimeError('Frozen vision source changed')
    rows=json.loads((out/'detector_frames.json').read_text())
    events=associate(rows)
    normal=[r for r in rows if not r['is_partial']]
    gaps=[dict(frame_index=r['frame_index'],seconds=r['elapsed_s']-rows[i-1]['elapsed_s'])
          for i,r in enumerate(rows) if i and r['elapsed_s']-rows[i-1]['elapsed_s']>1.5]
    summary=dict(frames_processed=len(rows),normal_nonpartial=statistics(normal),
        all_frames_including_partial=statistics(rows),windows_excluding_partial=windows(rows),
        final_complete=next(r for r in rows if r['product_stem']==COMPLETE),
        final_partial=next(r for r in rows if r['product_stem']==PARTIAL),
        association_is_image_only=True,identity_established=False,
        true_body_switch_count=None,selection_rank_changes_are_not_body_switches=True,
        cadence=dict(min_s=min(b['elapsed_s']-a['elapsed_s'] for a,b in zip(rows,rows[1:])),
            max_s=max(b['elapsed_s']-a['elapsed_s'] for a,b in zip(rows,rows[1:])),gaps_over_1_5s=gaps),
        frozen=config,scientific_boundary=WARNING,
        unavailable_metrics=['precision','recall','mAP','target identification accuracy','guidance accuracy'],
        limitations=['No exhaustive target annotations or semantic identity',
            'Candidate radius is optical component area, not physical radius or calibrated range',
            'No pixel validity mask beyond frozen finite/sentinel logic; black browse padding remains input',
            'PDS Bottom to Top convention not transformed; pixel-only coordinates as stored',
            'Image-only constant-velocity predictor can retain stale state when LOST',
            'No true physical body-switch metric; area rank can change without identity switching'])
    save(out/'frames.json',rows)
    table(out/'frames.csv',[{k:v for k,v in r.items() if k!='candidates'} for r in rows])
    table(out/'association_events.csv',events)
    save(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('frames_processed','identity_established')},indent=2))
    print(json.dumps(summary['normal_nonpartial'],indent=2))


def overlay(row, width=640):
    cv=require_opencv()
    image=cv.imread(row['png_local_path'],cv.IMREAD_UNCHANGED)
    # Draw on a new display canvas from original intensities, after inference.
    canvas=cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    for i,c in enumerate(row['candidates']):
        color=(70,240,90) if i==row['selected_candidate_index'] else (255,190,60)
        x,y,w,h=c['bbox_xywh']; cv.rectangle(canvas,(x,y),(x+w-1,y+h-1),color,2)
        center=tuple(np.rint(c['centroid_px']).astype(int))
        cv.drawMarker(canvas,center,color,cv.MARKER_CROSS,14,2)
        cv.putText(canvas,str(i),(x,max(14,y-3)),cv.FONT_HERSHEY_SIMPLEX,.6,color,1,cv.LINE_AA)
    canvas=cv.resize(canvas,(width,width),interpolation=cv.INTER_AREA)
    panel=np.zeros((width+128,width,3),dtype=np.uint8)
    panel[:width]=canvas
    text=[WARNING,f"{row['acquisition_utc']}  T{row['relative_to_impact_s']:.3f}s",
          f"Frame {row['frame_index']} | candidates {row['candidate_count']} | {row['track_status']}",
          f"Selected candidate: {row['selected_candidate_index']} (identity unverified)",
          'PARTIAL FINAL FRAME - separate analysis' if row['is_partial'] else row['product_stem']]
    for j,line in enumerate(text):
        cv.putText(panel,line,(8,width+20+j*23),cv.FONT_HERSHEY_SIMPLEX,.43,
            (50,150,255) if row['is_partial'] else (220,220,220),1,cv.LINE_AA)
    return panel


def render(out):
    cv=require_opencv()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((out/'frames.json').read_text()); summary=json.loads((out/'summary.json').read_text())
    directory=out/'representatives'; directory.mkdir(exist_ok=False)
    targets=['2022-09-26T22:00:00Z','2022-09-26T22:24:00Z','2022-09-26T22:45:00Z',
        '2022-09-26T23:00:00Z','2022-09-26T23:10:00Z','2022-09-26T23:13:24.183Z',
        '2022-09-26T23:14:14.183Z','2022-09-26T23:14:22.365Z','2022-09-26T23:14:23.328Z']
    chosen=[(name,min(rows,key=lambda r:abs((utc(r['acquisition_utc'])-utc(name)).total_seconds()))) for name in targets]
    for event in ('ACQUIRED','ENTERED_LOST','REACQUIRED'):
        match=next((r for r in rows if event in r['events']),None)
        if match: chosen.append((event,match))
    for name,predicate in [('no_candidate',lambda r:r['candidate_count']==0),
            ('multi_candidate',lambda r:r['candidate_count']>1),
            ('selected_candidate',lambda r:r['selected_candidate_index'] is not None)]:
        match=next((r for r in rows if predicate(r)),None)
        if match: chosen.append((name,match))
    manifest=[]
    for j,(reason,row) in enumerate(chosen):
        name=f"{j:02d}_{row['frame_index']:04d}.png"
        if not cv.imwrite(str(directory/name),overlay(row,768)): raise OSError(name)
        manifest.append(dict(reason=reason,frame_index=row['frame_index'],product_stem=row['product_stem'],path=str(directory/name)))
    save(out/'representatives.json',manifest)
    panels=[overlay(row,480) for _,row in chosen[:9]]
    sheet=np.vstack([np.hstack(panels[i:i+3]) for i in (0,3,6)])
    if not cv.imwrite(str(out/'contact_sheet.png'),sheet): raise OSError('Contact sheet')
    t=np.array([r['relative_to_impact_s']/60 for r in rows]); normal=np.array([not r['is_partial'] for r in rows])
    fig,axes=plt.subplots(4,1,figsize=(13,13),constrained_layout=True)
    axes[0].scatter(t[normal],[r['candidate_count'] for r in rows if not r['is_partial']],s=3,label='Nonpartial frames')
    axes[0].scatter(t[~normal],[r['candidate_count'] for r in rows if r['is_partial']],marker='x',color='red',label='Partial final')
    axes[0].set_ylabel('Candidate count'); axes[0].legend()
    axes[1].scatter([r['relative_to_impact_s']/60 for r in rows for c in r['candidates']],
        [c['radius_px'] for r in rows for c in r['candidates']],s=2,color='gray',alpha=.3,label='All components')
    selected=[r for r in rows if r['selected_candidate_index'] is not None]
    axes[1].scatter([r['relative_to_impact_s']/60 for r in selected],[r['selected_radius_px'] for r in selected],s=4,label='Selected candidate')
    axes[1].set_ylabel('Optical area radius (px)'); axes[1].set_yscale('log'); axes[1].legend()
    for k,coordinate in enumerate(('u','v')):
        axes[2].scatter([r['relative_to_impact_s']/60 for r in selected],[r['selected_centroid_px'][k] for r in selected],s=3,label=coordinate)
    axes[2].set_ylabel('Selected centroid (px)'); axes[2].legend()
    axes[3].scatter(t,[int(r['selected_candidate_index'] is not None) for r in rows],s=3,label='Selected')
    axes[3].scatter(t,[int(r['candidate_count']>0)*.4 for r in rows],s=2,label='Any candidate (0/0.4)')
    axes[3].set_ylabel('Availability'); axes[3].legend()
    for a in axes: a.grid(alpha=.25); a.set_xlabel('Minutes relative to approximate impact (negative before)')
    fig.suptitle(WARNING+'\nComponent measurements and selected track; body identity unverified')
    fig.savefig(out/'diagnostics.png',dpi=130); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    ax.hist([r['detector_wall_s']*1000 for r in rows],bins=50)
    ax.set(xlabel='Detector wall time (ms; excludes image read/render)',ylabel='Frames',title='Frozen detector processing time')
    fig.savefig(out/'processing_time.png',dpi=130);plt.close(fig)
    video=out/'recorded_draco_replay.mp4'
    if video.exists(): raise FileExistsError(video)
    writer=cv.VideoWriter(str(video),cv.VideoWriter_fourcc(*'mp4v'),60.,(640,768))
    if not writer.isOpened(): raise RuntimeError('Review MP4 encoder unavailable')
    try:
        for i,row in enumerate(rows):
            writer.write(overlay(row))
            if i%500==0: print(f'render {i}/{len(rows)}',flush=True)
    finally: writer.release()
    capture=cv.VideoCapture(str(video)); count=int(capture.get(cv.CAP_PROP_FRAME_COUNT)); fps=capture.get(cv.CAP_PROP_FPS)
    decoded=0
    while True:
        ok,frame=capture.read()
        if not ok: break
        if frame.shape[:2]!=(768,640): raise ValueError('Video dimensions changed')
        decoded+=1
    capture.release()
    if count!=len(rows) or decoded!=len(rows) or video.stat().st_size==0: raise RuntimeError('Video validation failed')
    save(out/'render_summary.json',dict(video_frames=count,decoded_frames=decoded,fps=fps,
        playback_s=count/fps,physical_span_s=rows[-1]['elapsed_s'],
        mean_acceleration_factor=rows[-1]['elapsed_s']/(count/fps),
        timing='Uniform one stored frame per video frame; acquisition cadence varies, UTC overlay authoritative',
        video_bytes=video.stat().st_size,video_sha256=digest(video),no_candidate_example_available=any(r['candidate_count']==0 for r in rows)))


def verify(out):
    baseline=json.loads((out/'preservation_before.json').read_text())
    changed=[];missing=[]
    for name,sha in baseline.items():
        path=ROOT/name
        if not path.is_file(): missing.append(name)
        elif digest(path)!=sha: changed.append(name)
    index=json.loads((ROOT/'data/draco/draco_terminal_to_impact_manifest.json').read_text())
    mismatched=[]
    for row in index:
        for kind in ('png','xml'):
            if digest(row[kind+'_local_path'])!=row[kind+'_sha256']: mismatched.append(row[kind+'_local_path'])
    result=dict(baseline_files=len(baseline),unchanged_files=len(baseline)-len(changed)-len(missing),
        changed=changed,missing=missing,original_pairs_checked=len(index),original_files_checked=len(index)*2,
        manifest_hash_mismatches=mismatched,passed=not(changed or missing or mismatched),
        no_aws_calls=True,no_uploads=True,no_scientific_changes=True,no_task3b=True)
    save(out/'preservation_verification.json',result)
    print(json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('Source preservation check failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['detect','analyze','render','verify'],required=True)
    parser.add_argument('--index',type=Path,default=ROOT/'data/draco/draco_terminal_to_impact_manifest.json')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.phase=='detect': detect(args.index,args.out)
    elif args.phase=='analyze': analyze(args.out)
    elif args.phase=='render': render(args.out)
    else: verify(args.out)
