"""Small judge interface with a global OS lock and one isolated simulation."""
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for
from .web_runner import PRESETS, ROOT, source_digest, write_json

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
ARTIFACT = re.compile(r"^(?:camera_(?:[0-9]{3}|final)\.png|telemetry\.csv|result\.json)$")


class Busy(Exception):
    pass


class JobManager:
    def __init__(self, root, timeout=180, max_jobs=20, ttl=86400, disk_cap=1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.max_jobs, self.ttl, self.disk_cap = timeout, max_jobs, ttl, disk_cap
        self.mutex = threading.Lock()
        self.submissions = deque()
        self.threads = []

    def acquire(self):
        handle = (self.root / ".worker.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise Busy("Another simulation is running") from None
        return handle

    def read(self, job_id):
        if not JOB_ID.fullmatch(job_id):
            return None
        path = self.root / job_id / "status.json"
        if not path.is_file() or time.time()-path.stat().st_mtime > self.ttl:
            return None
        status = json.loads(path.read_text())
        if status["state"] == "RUNNING":
            try:
                handle = self.acquire()
            except Busy:
                return status
            try:
                status["state"] = "COMPLETE" if (path.parent / "result.json").is_file() else "FAILED"
                if status["state"] == "FAILED":
                    status["message"] = "Worker interrupted; submit a new experiment"
                write_json(path, status)
            finally:
                handle.close()
        return status

    def housekeeping(self):
        while True:
            time.sleep(300)
            try:
                with self.mutex:
                    handle = self.acquire()
                    try:
                        self.prune()
                    finally:
                        handle.close()
            except (Busy, OSError):
                continue

    def prune(self):
        jobs = sorted((p for p in self.root.iterdir() if p.is_dir() and JOB_ID.fullmatch(p.name)),
                      key=lambda p: p.stat().st_mtime)
        def size():
            return sum(f.stat().st_size for d in jobs for f in d.rglob("*") if f.is_file())
        total = size()
        while jobs and (len(jobs) >= self.max_jobs or time.time()-jobs[0].stat().st_mtime > self.ttl or total > self.disk_cap):
            old = jobs.pop(0)
            total -= sum(f.stat().st_size for f in old.rglob("*") if f.is_file())
            # All paths originate beneath our exclusive result directory.
            shutil.rmtree(old)

    def submit(self, preset):
        if preset not in PRESETS:
            raise ValueError("Unknown preset")
        with self.mutex:
            handle = self.acquire()
            try:
                now = time.monotonic()
                while self.submissions and now-self.submissions[0] > 3600:
                    self.submissions.popleft()
                if len(self.submissions) >= 20:
                    raise Busy("Hourly demonstration limit reached; please try later")
                self.prune()
                job_id = uuid.uuid4().hex
                directory = self.root / job_id
                directory.mkdir()
                status = dict(id=job_id, preset=preset, state="RUNNING",
                              submitted_utc=datetime.now(timezone.utc).isoformat())
                write_json(directory / "status.json", status)
                thread = threading.Thread(target=self._execute, args=(directory, status, handle), daemon=True)
                thread.start()
                self.threads = [t for t in self.threads if t.is_alive()] + [thread]
                self.submissions.append(now)
                return status
            except BaseException:
                handle.close()
                raise

    def _execute(self, directory, status, handle):
        env = os.environ.copy()
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "1"
        env["MPLBACKEND"] = "Agg"
        try:
            with (directory / "worker.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "opendart.web_runner", "--preset", status["preset"],
                                "--out", str(directory)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout, check=True,
                               **({"pass_fds": (handle.fileno(),)} if os.name == "posix" else {}))
            if not (directory / "result.json").is_file():
                raise RuntimeError("Worker produced no result")
            status["state"] = "COMPLETE"
        except subprocess.TimeoutExpired:
            status.update(state="TIMEOUT", message="Cloud computation exceeded its 180-second limit")
        except Exception:
            status.update(state="FAILED", message="The simulation worker failed; no result is claimed")
        finally:
            status["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(directory / "status.json", status)
            handle.close()


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=1024, JOB_ROOT=os.getenv("OPENDART_RESULTS", str(ROOT / "results/web_live")))
    if config:
        app.config.update(config)
    manager = JobManager(app.config["JOB_ROOT"], timeout=app.config.get("JOB_TIMEOUT", 180))
    if not app.config.get("TESTING"):
        threading.Thread(target=manager.housekeeping, daemon=True).start()
    app.extensions["jobs"] = manager
    from .draco_web import register_draco
    register_draco(app, manager, Busy)
    fingerprint = source_digest()

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return render_template("index.html", presets=PRESETS)

    @app.get("/healthz")
    def health():
        return jsonify(status="ok", source_sha256=fingerprint,
                       release_sha256=os.getenv("OPENDART_RELEASE_SHA256", "local"))

    @app.post("/api/jobs")
    def submit():
        # Same-origin browser interaction only; no public arbitrary configuration.
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return jsonify(error="Cross-site submission is disabled"), 403
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return jsonify(error="Origin mismatch"), 403
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"preset"} or data["preset"] not in PRESETS:
            return jsonify(error="Choose one of the four fixed presets"), 400
        try:
            status = manager.submit(data["preset"])
        except Busy as exc:
            return jsonify(error=str(exc), state="BUSY"), 409
        return jsonify(**status, page=url_for("job_page", job_id=status["id"])), 202

    @app.get("/jobs/<job_id>")
    def job_page(job_id):
        status = manager.read(job_id)
        if status is None:
            abort(404)
        return render_template("job.html", job=status)

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        status = manager.read(job_id)
        if status is None:
            abort(404)
        progress = manager.root / job_id / "progress.json"
        if progress.is_file():
            status["progress"] = json.loads(progress.read_text())
        if status["state"] == "COMPLETE":
            status["result"] = json.loads((manager.root / job_id / "result.json").read_text())
        return jsonify(status)

    @app.get("/jobs/<job_id>/artifacts/<filename>")
    def artifact(job_id, filename):
        if not manager.read(job_id) or not ARTIFACT.fullmatch(filename):
            abort(404)
        return send_from_directory(manager.root / job_id, filename)

    @app.get("/evidence/<filename>")
    def evidence(filename):
        allowed = {"CAMPAIGN_REPORT.md", "groups.json", "success_rates.png", "uncertainty_diagnostics.png",
                   "loss_recovery.png", "representative_camera_sheet.png"}
        if filename not in allowed:
            abort(404)
        return send_from_directory(ROOT / "evidence", filename)

    return app
