"""Allowlisted DRACO presentation and one isolated live inference request."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from flask import abort,jsonify,render_template,request,send_from_directory
from .web_runner import ROOT

MEDIA={'replay.mp4','heldout_benchmark.png','comparison_compact.png','comparison_resolved.png','benchmark_metrics.json','manifest.json'}


def register_draco(app,manager,busy_error):
    assets=Path(app.config.get('DRACO_ROOT',os.getenv('OPENDART_DRACO_ROOT',str(ROOT/'draco_assets'))))
    catalog=json.loads((assets/'manifest.json').read_text()) if (assets/'manifest.json').is_file() else None
    choices={} if catalog is None else {r['id']:r for r in catalog['images']}

    @app.get('/draco')
    def draco_page():
        if catalog is None:abort(503,description='Curated DRACO evidence is not installed')
        metrics=json.loads((assets/'benchmark_metrics.json').read_text())
        return render_template('draco.html',catalog=catalog,metrics=metrics)

    @app.get('/draco/images/<image_id>.png')
    def draco_original(image_id):
        if image_id not in choices:abort(404)
        return send_from_directory(assets/'images',choices[image_id]['filename'])

    @app.get('/draco/media/<filename>')
    def draco_media(filename):
        if filename not in MEDIA or catalog is None:abort(404)
        return send_from_directory(assets,filename,conditional=True)

    @app.post('/api/draco/analyze')
    def draco_analyze():
        if request.headers.get('Sec-Fetch-Site')=='cross-site':return jsonify(error='Cross-site submission is disabled'),403
        origin=request.headers.get('Origin')
        if origin and origin.rstrip('/')!=request.host_url.rstrip('/'):return jsonify(error='Origin mismatch'),403
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)!={'image_id'} or not isinstance(data['image_id'],str) or data['image_id'] not in choices:
            return jsonify(error='Choose one of the curated DRACO frames'),400
        with manager.mutex:
            try:handle=manager.acquire()
            except busy_error:return jsonify(error='Another computation is running; try again shortly'),409
            now=time.monotonic()
            while manager.submissions and now-manager.submissions[0]>3600:manager.submissions.popleft()
            if len(manager.submissions)>=20:
                handle.close();return jsonify(error='Hourly demonstration limit reached'),429
            manager.submissions.append(now)
        started=time.perf_counter();request_id=uuid.uuid4().hex
        try:
            with tempfile.TemporaryDirectory(prefix='draco-',dir=manager.root) as directory:
                output=Path(directory)/'result.json';env=os.environ.copy()
                for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                env['PYTHONDONTWRITEBYTECODE']='1'
                subprocess.run([sys.executable,'-m','opendart.draco_web_runner','--assets',str(assets),
                    '--image-id',data['image_id'],'--request-id',request_id,'--out',str(output)],cwd=ROOT,env=env,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=True,
                    **({'pass_fds':(handle.fileno(),)} if os.name=='posix' else {}))
                result=json.loads(output.read_text())
            result['request_wall_ms']=(time.perf_counter()-started)*1000
            return jsonify(result)
        except subprocess.TimeoutExpired:return jsonify(error='Live inference exceeded its 15-second limit; no result is claimed'),504
        except Exception:return jsonify(error='Live inference failed verification; no result is claimed'),503
        finally:handle.close()
