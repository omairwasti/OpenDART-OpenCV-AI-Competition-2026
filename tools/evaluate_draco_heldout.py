"""One-shot held-out evaluation of unchanged, already recorded predictions.

This is an evaluation/presentation harness, not an inference implementation.
It never calls the detector/tracker or changes their frozen configuration.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import draco_development as dev
import prepare_draco_review as review

ROOT=dev.io.ROOT
PARENT=ROOT/'results/draco_specific_perception_2026-10-26'
DEVELOPMENT=PARENT/'development_01'
OUT=PARENT/'heldout_01'
LABEL='FINAL HELD-OUT EVALUATION - FROZEN METHOD - NO RETUNING'


def verify_freeze(directory=DEVELOPMENT,root=ROOT):
    freeze=json.loads((directory/'FREEZE.json').read_text())
    for name,sha in freeze['source_hashes'].items():
        if dev.io.digest(root/name)!=sha:raise RuntimeError('Frozen source mismatch: '+name)
    if dev.io.digest(directory/'frozen_configuration.json')!=freeze['configuration_sha256']:
        raise RuntimeError('Frozen configuration mismatch')
    return freeze


def reviewed_only(rows,benchmark):
    if any(r.get('split')!='held_out' for r in rows):raise ValueError('Only HELD-OUT rows are allowed')
    valid=review.validate_reviews(rows,[r for r in benchmark if r['split']=='held_out'])
    if any(r['is_partial'] for r in valid):raise ValueError('Partial is not an ordinary held-out sample')
    if not valid:raise ValueError('No genuinely reviewed held-out labels')
    return valid


def measure(annotations,predictions,config,designation=None):
    if any(r['review_status']!='HUMAN_REVIEWED' or r['split']!='held_out' for r in annotations):
        raise ValueError('Metric input must be reviewed HELD-OUT annotations only')
    records,summary=dev.evaluate(annotations,predictions,config,designation)
    summary['label']=LABEL
    summary['p90_centroid_error_px']=None
    summary['p90_note']='Omitted: at most eight compact labels, with fewer selected measurements; exact errors and medians are provided.'
    for r in records:r['label']=LABEL
    return records,summary


def frozen_baseline(rows):
    result={}
    for r in rows:
        i=r['selected_candidate_index'];c=None if i is None else r['candidates'][i]
        result[r['frame_index']]=dict(state=r['track_status'],selection_available=c is not None,
            target_center_uv=None if c is None else c['centroid_px'],
            visible_region_xywh=None if c is None else c['bbox_xywh'])
    return result


def comparison_rows(summaries,development=DEVELOPMENT):
    rows=[]
    files={'frozen_baseline':'frozen_baseline_development_metrics.json',
           'autonomous':'autonomous_development_metrics.json',
           'human_designated':'human_designated_development_metrics.json'}
    for name,held in summaries.items():
        old=json.loads((development/files[name]).read_text())
        for kind,metric,nkey in [('compact','within_5px_n','n'),('regions','iou_ge_0_3_n','n')]:
            a=old[kind];b=held[kind]
            rows.append(dict(method=name,metric=metric,development_numerator=a[metric],development_n=a[nkey],
                heldout_numerator=b[metric],heldout_n=b[nkey],
                difference_percentage_points=100*(b[metric]/b[nkey]-a[metric]/a[nkey]) if a[nkey] and b[nkey] else None))
        for kind,metric,nkey in [('compact','centroid_error_median_px','centroid_error_n'),('regions','median_iou','n')]:
            a=old[kind];b=held[kind]
            rows.append(dict(method=name,metric=metric,development_value=a[metric],development_n=a[nkey],
                heldout_value=b[metric],heldout_n=b[nkey],difference=b[metric]-a[metric]
                if a[metric] is not None and b[metric] is not None else None))
    return rows


def evaluate(out):
    # Verify every locked dependency BEFORE labels are consumed for scoring.
    freeze=verify_freeze()
    if (out/'EVALUATION_STARTED.json').exists():raise FileExistsError('One-shot evaluation already started; preserve and report its results')
    benchmark=json.loads((PARENT/'benchmark.json').read_text())
    export=json.loads((out/'original_held_out_export.json').read_text())
    annotations=reviewed_only(export,benchmark)
    normalized=json.loads((out/'validated_held_out_annotations.json').read_text())
    if annotations!=normalized:raise ValueError('Normalized labels differ from original reviewed entries')
    cfg=json.loads((DEVELOPMENT/'frozen_configuration.json').read_text())
    if cfg['designation']['frame_index']!=3825 or cfg['designation']['center_uv']!=[493.255,287.755]:
        raise ValueError('Unexpected frozen designation')
    if cfg['parameters']['point_evaluation_tolerance_px']!=5 or cfg['parameters']['region_evaluation_iou_threshold']!=.3:
        raise ValueError('Unexpected frozen metric definitions')
    input_paths={'held_out_export':out/'original_held_out_export.json','validated_annotations':out/'validated_held_out_annotations.json',
        'baseline_predictions':ROOT/'results/draco_frozen_replay_2026-10-26/frames.json',
        'draco_predictions':DEVELOPMENT/'qualitative_frozen_frames.json',
        'frozen_configuration':DEVELOPMENT/'frozen_configuration.json'}
    inputs={name:dict(path=str(path),sha256=dev.io.digest(path)) for name,path in input_paths.items()}
    dev.io.save(out/'EVALUATION_STARTED.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        label=LABEL,source_hashes=freeze['source_hashes'],inputs=inputs,evaluation_harness_sha256=dev.io.digest(__file__),
        inference_rerun=False,retuning_permitted=False))
    baseline_rows=json.loads(input_paths['baseline_predictions'].read_text())
    recorded=json.loads(input_paths['draco_predictions'].read_text())
    if len(recorded)!=4000 or len(baseline_rows)!=4000:raise ValueError('Frozen predictions incomplete')
    for a in annotations:
        for rows in (recorded,baseline_rows):
            r=rows[a['frame_index']]
            if r['product_stem']!=a['product_stem'] or r['acquisition_utc']!=a['acquisition_utc']:
                raise ValueError('Frozen prediction/annotation identity mismatch')
    predictions={'frozen_baseline':frozen_baseline(baseline_rows),
        **{mode:{r['frame_index']:r[mode] for r in recorded} for mode in ('autonomous','human_designated')}}
    summaries={};records={};click=cfg['designation']['frame_index']
    for name,prediction in predictions.items():
        per_frame,summary=measure(annotations,prediction,cfg['parameters'],click if name=='human_designated' else None)
        summaries[name]=summary;records[name]=per_frame
        dev.io.save(out/(name+'_heldout_metrics.json'),summary)
        dev.io.table(out/(name+'_heldout_samples.csv'),per_frame)
        dev.io.save(out/(name+'_heldout_samples.json'),per_frame)
        if name!='human_designated':
            _,matched=measure(annotations,prediction,cfg['parameters'],click)
            dev.io.save(out/(name+'_post_designation_heldout_metrics.json'),matched)
    dev.io.save(out/'heldout_comparison.json',summaries)
    comparison=comparison_rows(summaries)
    dev.io.save(out/'development_vs_heldout.json',comparison);dev.io.table(out/'development_vs_heldout.csv',comparison)
    dev.io.save(out/'EVALUATION_COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),label=LABEL,
        reviewed_heldout_frames=len(annotations),unreviewed_excluded=20-len(annotations),
        source_hashes_unchanged=verify_freeze()==freeze,configuration_unchanged=True,inference_rerun=False,
        original_heldout_export_sha256=dev.io.digest(PARENT/'draco_held_out_annotations.json'),
        reused_replay=str(DEVELOPMENT/'draco_specific_recorded_replay.mp4'),heldout_results_are_final=True))
    print(json.dumps(summaries,indent=2))


def render(out):
    """Annotation overlays only; no inference, original files remain unchanged."""
    cv=dev.method.require_opencv()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    annotations=json.loads((out/'validated_held_out_annotations.json').read_text())
    old=json.loads((ROOT/'results/draco_frozen_replay_2026-10-26/frames.json').read_text())
    new=json.loads((DEVELOPMENT/'qualitative_frozen_frames.json').read_text())
    pred={'frozen_baseline':frozen_baseline(old),**{m:{r['frame_index']:r[m] for r in new} for m in ('autonomous','human_designated')}}
    directory=out/'representatives';directory.mkdir(exist_ok=False)
    manifests=[];all_panels=[]
    for a in annotations:
        panels=[]
        for mode in pred:
            image=cv.imread(a['png_local_path'],cv.IMREAD_UNCHANGED);canvas=cv.cvtColor(image,cv.COLOR_GRAY2BGR);p=pred[mode][a['frame_index']]
            if a['target_center_uv']:
                u,v=a['target_center_uv'];cv.drawMarker(canvas,tuple(np.rint([u,v]).astype(int)),(80,255,80),cv.MARKER_CROSS,15,2);extent=180
            elif a['target_visible_box_xywh']:
                x,y,w,h=a['target_visible_box_xywh'];u,v=x+w/2,y+h/2;extent=min(1024,max(180,int(max(w,h)*1.4)))
                cv.rectangle(canvas,(round(x),round(y)),(round(x+w),round(y+h)),(80,255,80),2)
            else:u=v=512;extent=1024
            if p['visible_region_xywh']:
                x,y,w,h=p['visible_region_xywh'];cv.rectangle(canvas,(round(x),round(y)),(round(x+w),round(y+h)),(0,160,255),2)
            x0=max(0,min(1024-extent,int(u-extent/2)));y0=max(0,min(1024-extent,int(v-extent/2)))
            panel=np.zeros((450,400,3),np.uint8);panel[:400]=cv.resize(canvas[y0:y0+extent,x0:x0+extent],(400,400),interpolation=cv.INTER_NEAREST)
            cv.putText(panel,f"{a['frame_index']} {mode} | {p['state']}",(5,419),cv.FONT_HERSHEY_SIMPLEX,.35,(235,235,235),1)
            cv.putText(panel,'HELD-OUT | green reviewed / orange prediction',(5,442),cv.FONT_HERSHEY_SIMPLEX,.34,(235,235,235),1)
            panels.append(panel)
        name=f"heldout_{a['frame_index']:04d}.png";combined=np.hstack(panels)
        if not cv.imwrite(str(directory/name),combined):raise OSError(name)
        manifests.append(dict(frame_index=a['frame_index'],product_stem=a['product_stem'],class_name=a['label_class'],path=str(directory/name),
            display_only_crop=True,no_inference=True))
        all_panels.append(combined)
    dev.io.save(out/'representatives.json',manifests)
    # Fixed chronological representatives include early misses, points, regions.
    selected=[i for i,a in enumerate(annotations) if a['frame_index'] in (717,1667,2293,2995,3763,3845,3990)]
    if not cv.imwrite(str(out/'heldout_contact_sheet.png'),np.vstack([all_panels[i] for i in selected])):raise OSError('Contact sheet')
    summary=json.loads((out/'heldout_comparison.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    names=['frozen_baseline','autonomous']
    axes[0].bar(['Frozen','DRACO auto'],[summary[n]['compact']['within_5px_n'] for n in names],color=['gray','tab:blue'])
    axes[0].set(ylabel='Point matches within 5px (n=8)',ylim=(0,8))
    for name,color in [('frozen_baseline','gray'),('autonomous','tab:blue'),('human_designated','tab:orange')]:
        with (out/(name+'_heldout_samples.csv')).open(newline='') as stream:rows=[r for r in csv.DictReader(stream) if r['eligible']=='True' and r['class_name']=='DIMORPHOS_RESOLVED_BODY']
        axes[1].scatter([int(r['frame_index']) for r in rows],[float(r['region_iou']) for r in rows],label=f'{name} (n={len(rows)})',color=color,s=30)
    axes[1].set(xlabel='Reviewed held-out source frame',ylabel='Visible-region IoU',ylim=(-.03,1.03));axes[1].legend(fontsize=7)
    oldmetrics=json.loads((DEVELOPMENT/'autonomous_development_metrics.json').read_text())
    axes[2].bar(['Dev 8/9','Held-out '+str(summary['autonomous']['compact']['within_5px_n'])+'/8'],
        [oldmetrics['compact']['within_5px_n']/9,summary['autonomous']['compact']['within_5px_n']/8],color=['tab:blue','tab:green'])
    axes[2].set(ylabel='Autonomous point match fraction',ylim=(0,1))
    fig.suptitle('FINAL HELD-OUT EVALUATION - FROZEN METHOD\n18 reviewed / 20 sampled; two UNREVIEWED boundary-clipped cases excluded from all metrics')
    fig.savefig(out/'heldout_benchmark.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['evaluate','render','verify'],required=True)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    {'evaluate':evaluate,'render':render,'verify':dev.io.verify}[args.phase](args.out)
