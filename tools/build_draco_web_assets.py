"""Build a small curated release with strictly causal, validated checkpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import evaluate_draco_heldout as held
from opendart.draco_web_runtime import load_method,snapshot,restore,sha,METHOD_SHA,PROFILE_SHA

INDICES=(0,1291,2419,3226,3764,3825,3946,3973,3991,3994,3998,3999)
DESCRIPTIONS=('Early faint system; the frozen autonomous method has no selected target.',
 'Small binary imagery; autonomous target acquisition remains unresolved.',
 'Separated compact components during the terminal approach.',
 'Mid approach with a small companion region.',
 'Beginning of the FINAL collection; clearly separated bodies.',
 'Early FINAL binary imagery; this live analyzer uses autonomous mode.',
 'Approximately one minute before impact; growing visible body.',
 'Approximately thirty seconds before impact; resolved target region.',
 'Approximately ten seconds before impact; surface detail and fragmentation.',
 'Resolved terrain approaching the available image boundary.',
 'Final complete image: visible terrain, not a complete isolated-body silhouette.',
 'Final partial transmission: only the received strip is visible; missing extent is not inferred.')


def build(output):
    held.verify_freeze();output=Path(output);output.mkdir(exist_ok=False)
    for sub in ('images','checkpoints'): (output/sub).mkdir()
    index=json.loads((ROOT/'data/draco/draco_terminal_to_impact_manifest.json').read_text())
    expected=json.loads((held.DEVELOPMENT/'qualitative_frozen_frames.json').read_text())
    module=load_method();cfg=module.load_config();feature=module.Features(cfg);track=module.Track(cfg,'autonomous');cv=module.require_opencv();cv.setNumThreads(1)
    epoch=datetime.fromisoformat(index[0]['acquisition_utc'].replace('Z','+00:00'));impact=datetime.fromisoformat('2022-09-26T23:14:24.183+00:00')
    catalog=[];validated=[]
    for i,item in enumerate(index):
        utc=datetime.fromisoformat(item['acquisition_utc'].replace('Z','+00:00'));elapsed=(utc-epoch).total_seconds();image=cv.imread(item['png_local_path'],cv.IMREAD_UNCHANGED)
        if i in INDICES:
            image_id=f'f{i:04d}';file=Path(item['png_local_path']).name;checkpoint=output/'checkpoints'/(image_id+'.npz')
            snapshot(feature,track,checkpoint);resumed_feature,resumed_track=restore(module,cfg,checkpoint,i)
            obs=resumed_feature.update(image,elapsed);resumed=resumed_track.update(obs,elapsed,i,item['is_partial'])
            reference=expected[i]['autonomous'];keys=('state','identity_status','target_center_uv','visible_region_xywh','selection_available','resolved_body','region_component_count','events')
            if any(resumed[k]!=reference[k] for k in keys):raise RuntimeError('Checkpoint does not reproduce frozen output: '+image_id)
            shutil.copyfile(item['png_local_path'],output/'images'/file)
            if sha(output/'images'/file)!=item['png_sha256']:raise RuntimeError('Original PNG changed')
            catalog.append(dict(id=image_id,frame_index=i,filename=file,product_id=item['product_stem'],acquisition_utc=item['acquisition_utc'],
                elapsed_s=elapsed,time_to_impact_s=(impact-utc).total_seconds(),mission_phase=item['mission_phase'],is_partial=item['is_partial'],
                interpretation=DESCRIPTIONS[INDICES.index(i)],sha256=item['png_sha256'],checkpoint_sha256=sha(checkpoint),
                archive_url=f"https://pds-smallbodies.astro.umd.edu/holdings/pds4-dart:data_dracocal-v3.0/{item['mission_phase']}/2022/269/"))
            validated.append(dict(id=image_id,frame_index=i,keys_verified=list(keys),state=resumed['state']))
        obs=feature.update(image,elapsed);track.update(obs,elapsed,i,item['is_partial'])
        if i%400==0:print('Checkpoint prefix',i,'/4000',flush=True)
    keys=('state','identity_status','target_center_uv','visible_region_xywh','selection_available','resolved_body','region_component_count','events')
    (output/'checkpoints/expected.json').write_text(json.dumps({f'f{i:04d}':{k:expected[i]['autonomous'][k] for k in keys} for i in INDICES},indent=2)+'\n')
    original=held.DEVELOPMENT/'draco_specific_recorded_replay.mp4'
    expected_video=json.loads((held.DEVELOPMENT/'render_verification.json').read_text())
    if sha(original)!=expected_video['sha256']:raise RuntimeError('Frozen replay changed')
    # Browser-compatible presentation copy only: same 4,000 frames, no inference.
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(original),
        '-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(output/'replay.mp4')],check=True)
    capture=cv.VideoCapture(str(output/'replay.mp4'));frames=int(capture.get(cv.CAP_PROP_FRAME_COUNT));fps=capture.get(cv.CAP_PROP_FPS);capture.release()
    if frames!=4000 or abs(fps-60)>.001:raise RuntimeError('Presentation video changed timing')
    for source,dest in [(held.OUT/'heldout_benchmark.png','heldout_benchmark.png'),
        (held.OUT/'representatives/heldout_2293.png','comparison_compact.png'),(held.OUT/'representatives/heldout_3990.png','comparison_resolved.png')]:shutil.copyfile(source,output/dest)
    all_metrics=json.loads((held.OUT/'heldout_comparison.json').read_text())
    metrics={mode:{k:s[k] for k in ('compact','regions','uncertain','excluded_pre_or_at_designation')} for mode,s in all_metrics.items()}
    # Aggregate values only, no annotations/reviewer identities/local paths.
    (output/'benchmark_metrics.json').write_text(json.dumps(dict(reviewed=18,sampled=20,unreviewed_boundary_excluded=2,
        compact_labels=8,region_labels=8,uncertain_labels=2,modes=metrics),indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(images=catalog,method_sha256=METHOD_SHA,configuration_sha256=PROFILE_SHA,
        original_frozen_replay_sha256=expected_video['sha256'],browser_replay_sha256=sha(output/'replay.mp4'),
        live_method='Autonomous frozen checkpoint continuation; state saved strictly before selected image; no labels',
        media={name:sha(output/name) for name in ('replay.mp4','heldout_benchmark.png','comparison_compact.png','comparison_resolved.png','benchmark_metrics.json')}),indent=2)+'\n')
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total>50*1024**2:raise RuntimeError('Curated assets unexpectedly exceed 50 MiB')
    validation=dict(checkpoints=validated,frames_replayed_causally=4000,original_pngs=len(catalog),total_bytes=total,
        no_labels_in_checkpoints=True,video_frames=4000,video_fps=fps,original_video_unchanged=sha(original)==expected_video['sha256'])
    (output.parent/'checkpoint_validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);build(parser.parse_args().out)
