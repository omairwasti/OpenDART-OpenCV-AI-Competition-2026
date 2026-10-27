"""Public allowlist, real frozen inference and checkpoint semantic validation."""
import json
from pathlib import Path
import subprocess
import uuid

import pytest

from opendart.web import create_app,Busy
from opendart.web_runner import ROOT,PRESETS,preset_config
from opendart import draco_web_runtime as runtime


@pytest.fixture(scope='module')
def assets():
    root=ROOT/'draco_assets'
    if not root.is_dir():root=ROOT/'results/draco_web_task3a6_2026-10-26/curated'
    assert (root/'manifest.json').is_file(),'Build the curated assets before integration tests'
    return root


@pytest.fixture
def client(tmp_path,assets):
    return create_app(dict(TESTING=True,JOB_ROOT=str(tmp_path/'jobs'),DRACO_ROOT=str(assets))).test_client()


def test_draco_page_metrics_gallery_and_navigation(client,assets):
    page=client.get('/draco');assert page.status_code==200
    text=page.get_data(as_text=True)
    for phrase in ('Real DRACO','not closed-loop control','18 reviewed / 20 sampled','82.483','1.218','0.802','0.871','same six-frame subset','final ten seconds'):
        assert phrase in text
    catalog=runtime.manifest(assets)
    metrics=json.loads((assets/'benchmark_metrics.json').read_text())
    assert metrics['reviewed']==18 and metrics['sampled']==20 and metrics['unreviewed_boundary_excluded']==2
    assert metrics['modes']['autonomous']['compact']['within_5px_n']==6
    assert f"{metrics['modes']['autonomous']['compact']['centroid_error_median_px']:.3f} px, n=6" in text
    assert f"{metrics['modes']['autonomous']['regions']['median_iou']:.3f}, n=8" in text
    assert f"{metrics['modes']['human_designated']['regions']['median_iou']:.3f}, n=6" in text
    assert len(catalog['images'])==12
    for row in catalog['images']:
        response=client.get('/draco/images/'+row['id']+'.png')
        assert response.status_code==200
        import hashlib
        assert hashlib.sha256(response.data).hexdigest()==row['sha256']
    assert b'Real DRACO imagery' in client.get('/').data
    for preset in PRESETS:assert preset.encode() in client.get('/').data


@pytest.mark.parametrize('payload',[{},[],{'image_id':[]},{'image_id':'../../secrets'},
    {'image_id':'http://example.invalid/x.png'},{'image_id':'not-curated'},
    {'image_id':'f0000','threshold':0},{'image_id':'f0000','url':'file:///etc/passwd'}])
def test_public_analyzer_rejects_arbitrary_inputs(client,payload):
    assert client.post('/api/draco/analyze',json=payload).status_code==400


def test_private_checkpoint_and_paths_not_served(client):
    for path in ('/draco/media/checkpoints','/draco/media/expected.json','/draco/media/f0000.npz','/draco/images/unknown.png','/draco/media/../../../etc/passwd'):
        assert client.get(path).status_code==404
    assert client.post('/api/draco/analyze',json={'image_id':'f0000'},headers={'Origin':'https://evil.invalid'}).status_code==403
    assert client.post('/api/draco/analyze',json={'image_id':'f0000'},headers={'Sec-Fetch-Site':'cross-site'}).status_code==403
    assert client.get('/api/draco/analyze').status_code==405


@pytest.mark.parametrize('image_id',['f0000','f2419','f3991','f3999'])
def test_real_checkpoint_inference_reproduces_frozen_output(assets,image_id):
    result=runtime.analyze(assets,image_id,uuid.uuid4().hex)
    expected=json.loads((assets/'checkpoints/expected.json').read_text())[image_id]
    for key,value in expected.items():assert result[key]==value
    assert result['opencv']=='5.0.0' and result['inference_executed'] is True
    assert result['server_processing_ms']>0 and result['overlay_png_base64']
    if image_id=='f0000':assert result['visible_region_xywh'] is None and result['state']=='SEARCH_UNRESOLVED'
    if image_id=='f3991':assert result['state']=='RESOLVED_BODY' and result['representation']=='VISIBLE_FOREGROUND_ENVELOPE'
    if image_id=='f3999':assert result['state']=='PARTIAL_FRAME' and result['target_center_uv'] is None and not result['selection_available']


def test_every_curated_checkpoint_matches_not_just_representatives(assets):
    expected=json.loads((assets/'checkpoints/expected.json').read_text())
    for image_id,reference in expected.items():
        actual=runtime.analyze(assets,image_id,uuid.uuid4().hex)
        assert all(actual[k]==v for k,v in reference.items())


def test_http_analyzer_runs_new_isolated_frozen_worker(client,monkeypatch):
    actual=subprocess.run;calls=[]
    def observed(command,**kwargs):
        calls.append(command)
        assert command[1:3]==['-m','opendart.draco_web_runner'] and kwargs['timeout']==15
        return actual(command,**kwargs)
    monkeypatch.setattr(subprocess,'run',observed)
    first=client.post('/api/draco/analyze',json={'image_id':'f2419'})
    second=client.post('/api/draco/analyze',json={'image_id':'f2419'})
    assert first.status_code==second.status_code==200
    assert len(calls)==2 and first.json['request_id']!=second.json['request_id']
    assert first.json['inference_executed'] and second.json['inference_executed']
    assert first.json['target_center_uv']==second.json['target_center_uv']


def test_inference_really_calls_frozen_feature_update(assets,monkeypatch):
    original=runtime.load_method;calls=[]
    def observed_load():
        module=original();update=module.Features.update
        def observed_update(self,image,timestamp):
            calls.append((image.shape,timestamp));return update(self,image,timestamp)
        monkeypatch.setattr(module.Features,'update',observed_update);return module
    monkeypatch.setattr(runtime,'load_method',observed_load)
    runtime.analyze(assets,'f0000','unit-test')
    assert calls==[((1024,1024),0.0)]


def test_draco_and_synthetic_share_global_lock(client):
    manager=client.application.extensions['jobs'];handle=manager.acquire()
    try:
        assert client.post('/api/draco/analyze',json={'image_id':'f0000'}).status_code==409
        assert client.post('/api/jobs',json={'preset':'NOMINAL'}).status_code==409
    finally:handle.close()


def test_video_range_and_public_aggregate_metadata(client):
    response=client.get('/draco/media/replay.mp4',headers={'Range':'bytes=0-127'})
    assert response.status_code==206 and len(response.data)==128
    metadata=client.get('/draco/media/manifest.json').get_data(as_text=True)
    assert 'C:\\' not in metadata and '/Users/' not in metadata
    assert client.get('/draco/media/heldout_benchmark.png').status_code==200
