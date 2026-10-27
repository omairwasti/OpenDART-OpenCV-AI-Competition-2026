"""Bounded checkpoint continuation adapter; the DRACO algorithm is unmodified."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHOD_SHA='6afb56362fe841f2158c291600d63760929ea9948940d984dc05d9f58796aba9'
PROFILE_SHA='84554612ed68c6b699ac2d4ccc013a38dcf317d38eb95b4d9740f3e2ea7558d5'
VISION_SHA='ea6459012530a9f55e6c51a485bbdc217ea9a206c16f5624cc646b096c035aad'
TRACK_FIELDS=('used_designation','position','area','box','missed','state','identity','pending',
              'pending_count','ever','resolved_count','resolved','last_time')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_method():
    paths=[(ROOT/'tools/draco_perception.py',METHOD_SHA),
           (ROOT/'tools/draco_perception_config_v1.json',PROFILE_SHA),(ROOT/'opendart/vision.py',VISION_SHA)]
    if any(sha(path)!=expected for path,expected in paths):raise RuntimeError('Frozen perception verification failed')
    spec=importlib.util.spec_from_file_location('opendart_frozen_draco',paths[0][0]);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(feature,track,path):
    """Save only state strictly before the selected image; no pickle or labels."""
    metadata=dict(feature_frames=feature.frames,feature_last_time=feature.last_time,
                  track={key:(getattr(track,key).tolist() if isinstance(getattr(track,key),np.ndarray) else getattr(track,key)) for key in TRACK_FIELDS})
    if Path(path).exists():raise FileExistsError(path)
    np.savez_compressed(path,past=feature.past,previous=feature.previous if feature.previous is not None else np.empty(0,np.float32),
                        metadata=np.asarray(json.dumps(metadata)))


def restore(module,config,path,frame_index):
    feature=module.Features(config);track=module.Track(config,'autonomous')
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!={'past','previous','metadata'}:raise ValueError('Invalid checkpoint fields')
        metadata=json.loads(str(saved['metadata']));past=saved['past'];previous=saved['previous']
        if past.shape!=(1024,1024) or past.dtype!=np.float32:raise ValueError('Invalid persistence state')
        n=config['phase_size']
        if previous.shape not in ((0,),(n,n)) or previous.dtype!=np.float32:raise ValueError('Invalid previous-image state')
        if metadata['feature_frames']!=frame_index or set(metadata['track'])!=set(TRACK_FIELDS):raise ValueError('Checkpoint identity mismatch')
        feature.past=past.copy();feature.previous=previous.copy() if previous.size else None
        feature.frames=metadata['feature_frames'];feature.last_time=metadata['feature_last_time']
        for key,value in metadata['track'].items():setattr(track,key,np.asarray(value,float) if key in ('position','pending') and value is not None else value)
    return feature,track


def manifest(asset_root):
    return json.loads((Path(asset_root)/'manifest.json').read_text())


def analyze(asset_root,image_id,request_id):
    started=time.perf_counter();asset_root=Path(asset_root);catalog=manifest(asset_root)
    choices={r['id']:r for r in catalog['images']}
    if not isinstance(image_id,str) or image_id not in choices:raise ValueError('Choose an allowlisted DRACO frame')
    item=choices[image_id];module=load_method();cv=module.require_opencv()
    if cv.__version__!='5.0.0':raise RuntimeError('OpenCV 5.0.0 required')
    cv.setNumThreads(1)
    image_path=asset_root/'images'/item['filename'];checkpoint=asset_root/'checkpoints'/(image_id+'.npz')
    if sha(image_path)!=item['sha256'] or sha(checkpoint)!=item['checkpoint_sha256']:raise RuntimeError('Curated input verification failed')
    config=module.load_config();feature,track=restore(module,config,checkpoint,item['frame_index'])
    image=cv.imread(str(image_path),cv.IMREAD_UNCHANGED)
    observation=feature.update(image,item['elapsed_s'])
    result=track.update(observation,item['elapsed_s'],item['frame_index'],item['is_partial'])
    canvas=cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    if result['visible_region_xywh'] is not None:
        x,y,w,h=map(lambda v:int(round(v)),result['visible_region_xywh'])
        cv.rectangle(canvas,(x,y),(x+w-1,y+h-1),(70,210,255),2)
    if result['target_center_uv'] is not None:cv.drawMarker(canvas,tuple(np.rint(result['target_center_uv']).astype(int)),(70,210,255),cv.MARKER_CROSS,14,2)
    cv.putText(canvas,result['state'],(18,32),cv.FONT_HERSHEY_SIMPLEX,.7,(70,210,255),2,cv.LINE_AA)
    cv.putText(canvas,'RECORDED DRACO - NOT CLOSED LOOP',(18,60),cv.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1,cv.LINE_AA)
    ok,encoded=cv.imencode('.png',canvas)
    if not ok:raise RuntimeError('Overlay encoding failed')
    return dict(request_id=request_id,inference_executed=True,execution='LIVE_FROZEN_CHECKPOINT_CONTINUATION',
        image_id=image_id,product_id=item['product_id'],acquisition_utc=item['acquisition_utc'],time_to_impact_s=item['time_to_impact_s'],
        is_partial=item['is_partial'],**result,raw_candidate_count=observation['raw_candidate_count'],
        kept_candidate_count=len(observation['candidates']),opencv=cv.__version__,python=platform.python_version(),
        platform=platform.system(),server_processing_ms=(time.perf_counter()-started)*1000,
        method_sha256=METHOD_SHA,configuration_sha256=PROFILE_SHA,original_sha256=item['sha256'],
        checkpoint_sha256=item['checkpoint_sha256'],release_sha256=os.getenv('OPENDART_RELEASE_SHA256','local'),
        overlay_png_base64=base64.b64encode(encoded.tobytes()).decode('ascii'))
