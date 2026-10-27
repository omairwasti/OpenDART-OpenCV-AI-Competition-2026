"""Short orbit-context and closed-loop pixel-feedback movie; physical time is SI."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
from .model import Binary
from .planar import PlanarConfig, simulate_planar
from .vision import require_opencv, annotate_candidates

BG = (24, 18, 12)
INK = (232, 235, 239)
TEAL = (170, 215, 60)
GOLD = (70, 190, 250)


def text(frame, string, xy, size=.55, color=INK, thick=1):
    cv = require_opencv()
    cv.putText(frame, str(string), tuple(map(int, xy)), cv.FONT_HERSHEY_SIMPLEX,
               size, color, thick, cv.LINE_AA)


def clock_string(seconds):
    h = int(seconds//3600); m = int(seconds%3600//60); s = seconds%60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def world_panel(binary, time_s, spacecraft=None, estimate_sc=None, trail=None):
    cv = require_opencv()
    size = 650
    panel = np.full((size, size, 3), BG, np.uint8)
    p = binary.states(time_s)[0][:, :2]
    if spacecraft is None:
        center = np.zeros(2); span = 3400.
    else:
        center = (spacecraft+p[1])/2
        span = max(3400., 1.35*np.linalg.norm(spacecraft-p[1])+1000.)
    scale = (size-80)/span
    def xy(value):
        q = (np.asarray(value)-center)*scale
        return tuple(np.rint([size/2+q[0], size/2-q[1]]).astype(int))
    for gx in np.linspace(-span, span, 9):
        cv.line(panel, xy(center+[gx, -span]), xy(center+[gx, span]), (47, 41, 32), 1)
        cv.line(panel, xy(center+[-span, gx]), xy(center+[span, gx]), (47, 41, 32), 1)
    eta = binary.secondary_fraction
    for radius in [eta*binary.separation_m, (1-eta)*binary.separation_m]:
        points = np.array([xy(radius*np.array([np.cos(a), np.sin(a)]))
                           for a in np.linspace(0, 2*np.pi, 200)], np.int32)
        cv.polylines(panel, [points], True, (100, 82, 61), 1, cv.LINE_AA)
    if trail is not None and len(trail) > 1:
        points = np.array([xy(v) for v in trail], np.int32)
        cv.polylines(panel, [points], False, (130, 215, 75), 1, cv.LINE_AA)
    for body in (0, 1):
        angle = binary.phase_rad+(2*np.pi/binary.primary_spin_period_s if body == 0
                                  else binary.n)*time_s
        axes = tuple(max(3, int(a*scale)) for a in binary.axes_m[body, :2])
        cv.ellipse(panel, xy(p[body]), axes, -np.degrees(angle), 0, 360,
                   (135, 135, 140) if body == 0 else (185, 185, 190), -1, cv.LINE_AA)
        end = p[body]+.8*binary.axes_m[body,0]*np.array([np.cos(angle), np.sin(angle)])
        cv.line(panel, xy(p[body]), xy(end), (48, 48, 55), 2, cv.LINE_AA)
        label = "Didymos" if body == 0 else "Dimorphos"
        where = np.array(xy(p[body]))+[10, -max(axes)-10]
        text(panel, label, where, .47)
    if spacecraft is not None:
        cv.drawMarker(panel, xy(spacecraft), (130, 255, 100), cv.MARKER_TRIANGLE_UP, 15, 2)
        cv.line(panel, xy(spacecraft), xy(p[1]), (70, 100, 70), 1)
        text(panel, "spacecraft", np.array(xy(spacecraft))+[12, 20], .45, (130, 255, 100))
    if estimate_sc is not None:
        cv.drawMarker(panel, xy(estimate_sc), GOLD, cv.MARKER_CROSS, 14, 1)
    bar_m = 1000 if span < 6000 else 5000
    xbar, ybar = 35, size-35
    cv.line(panel, (xbar, ybar), (xbar+int(bar_m*scale), ybar), INK, 2)
    text(panel, f"{bar_m/1000:g} km", (xbar, ybar-9), .45)
    text(panel, "Physical world XY | view auto-zooms", (20, 28), .6, TEAL)
    return panel


def orbit_frame(cfg, binary, index):
    t = cfg.period_s*index/max(cfg.overview_frames-1, 1)
    frame = np.full((800, 1280, 3), BG, np.uint8)
    frame[80:730, 20:670] = world_panel(binary, t)
    text(frame, "OpenDART | binary motion before terminal approach", (25, 38), .82, TEAL, 2)
    text(frame, "ORBIT CONTEXT: kinematics only", (700, 120), .67, GOLD)
    lines = [
        f"Physical elapsed: {clock_string(t)}",
        "Dimorphos orbit: 11 h 55 min",
        "Didymos spin: 2.26 h",
        f"Dimorphos orbit angle: {360*t/cfg.period_s:.1f} deg",
        f"Didymos rotations: {t/cfg.spin_period_s:.2f}",
        "",
        "One physical orbit is sampled analytically.",
        "Only movie playback is accelerated.",
        "",
        "Next: camera pixels -> OpenCV -> EKF",
        "      -> guidance -> thrust -> next view.",
        "",
        "No spacecraft is propagated in this",
        "overview. The closed loop starts next.",
    ]
    for j, line in enumerate(lines):
        text(frame, line, (700, 168+31*j), .56)
    text(frame, "Physical periods retained. Short terminal phase follows.", (25, 775), .5)
    return frame


def sparkline(frame, rows, key, rect, label, color, limit=None):
    cv = require_opencv()
    x, y, w, h = rect
    cv.rectangle(frame, (x, y), (x+w, y+h), (74, 65, 51), 1)
    values = np.array([r[key] for r in rows], float)
    top = limit or max(float(np.nanmax(values)), 1e-6)
    ts = np.array([r["elapsed_s"] for r in rows])
    xx = x+np.clip(ts/650, 0, 1)*w
    yy = y+h-np.clip(values/top, 0, 1)*h
    if len(values) > 1:
        points = np.rint(np.c_[xx, yy]).astype(np.int32)
        cv.polylines(frame, [points], False, color, 1, cv.LINE_AA)
    text(frame, f"{label}: {values[-1]:.2f}", (x+7, y+19), .42, color)


def dashboard(cfg, binary, row, image, candidates, selected, rows):
    cv = require_opencv()
    frame = np.full((800, 1280, 3), BG, np.uint8)
    sc = np.array([row["spacecraft_x_m"], row["spacecraft_y_m"]])
    target = np.array([row["target_x_m"], row["target_y_m"]])
    estimate_sc = target-np.array([row["estimate_x_m"], row["estimate_y_m"]])
    trail = [np.array([r["spacecraft_x_m"], r["spacecraft_y_m"]]) for r in rows]
    frame[80:730, 20:670] = world_panel(binary, row["physical_time_s"], sc, estimate_sc, trail)
    frame[80:592, 720:1232] = cv.resize(annotate_candidates(image, candidates, selected), (512, 512))
    text(frame, "OpenDART | 2D pixel detection, estimation and interception", (25, 36), .78, TEAL, 2)
    text(frame, f"Terminal elapsed {row['elapsed_s']:6.1f} s | {row['mode']} | {row['tracker']}",
         (25, 65), .54, GOLD)
    text(frame, "Synthetic camera -> OpenCV detections", (727, 75), .52)
    text(frame, f"true range {row['true_range_m']:8.1f} m   estimated {row['estimated_range_m']:8.1f} m",
         (720, 617), .49)
    text(frame, f"image update {bool(row['accepted'])}   blobs {row['candidates']}   force {row['force_n']:.2f} N",
         (720, 642), .49)
    sparkline(frame, rows, "force_n", (720, 660, 242, 72), "Thrust N", TEAL, cfg.thrust_n)
    sparkline(frame, rows, "position_error_m", (985, 660, 242, 72), "Position error m", GOLD)
    size_label = "bearing + apparent size (known radius)" if cfg.use_apparent_size else "bearing only + range prior"
    text(frame, "Navigation: "+size_label, (25, 758), .52)
    text(frame, f"Orbit advance {360*row['elapsed_s']/cfg.period_s:.2f} deg | spin advance "
                f"{360*row['elapsed_s']/cfg.spin_period_s:.2f} deg | movie playback only is sped up",
         (25, 785), .48)
    return frame


class Movie:
    def __init__(self, out, fps, enabled=True):
        self.cv = require_opencv(); self.path = None; self.writer = None; self.count = 0
        if enabled:
            for filename, codec in [("simulation.mp4", "mp4v"), ("simulation.avi", "MJPG")]:
                path = out/filename
                writer = self.cv.VideoWriter(str(path), self.cv.VideoWriter_fourcc(*codec), fps, (1280, 800))
                if writer.isOpened():
                    self.path, self.writer = path, writer
                    break
                writer.release()
            if self.writer is None:
                raise RuntimeError("No usable MP4/AVI encoder; rerun with --no-video")

    def append(self, image):
        if self.writer:
            self.writer.write(image); self.count += 1

    def close(self):
        if self.writer: self.writer.release()


def save_outputs(result, out, movie, wall_s):
    cv = require_opencv(); out.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    summary.update(compute_wall_seconds=wall_s,
                   video_frames=movie.count, video_file=movie.path.name if movie.path else None,
                   video_playback_seconds=movie.count/result["config"]["fps"])
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out/"config.json").write_text(json.dumps(result["config"], indent=2)+"\n")
    (out/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, opencv=cv.__version__), indent=2)+"\n")
    rows = result["telemetry"]
    with (out/"telemetry.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cv.imwrite(str(out/"last_camera.png"), result["final_camera"])
    if result["final_dashboard"] is not None:
        cv.imwrite(str(out/"final_dashboard.png"), result["final_dashboard"])
    filename = movie.path.name if movie.path else ""
    video = f'<video controls width="100%" src="{filename}"></video>' if filename else ""
    page = f"""<!doctype html><html><meta charset="utf-8"><title>OpenDART 2D results</title>
    <style>body{{background:#101820;color:#e8edf2;font:17px sans-serif;max-width:1100px;margin:30px auto}}
    img,video{{max-width:100%}}a{{color:#75d9b7}}pre{{white-space:pre-wrap}}</style>
    <h1>OpenDART 2D image-feedback simulation</h1><p>{summary['status']} |
    OpenCV {summary['opencv_version']} | physical periods retained</p>{video}
    <p>The first scene explains one physical binary orbit. The second scene runs the causal
    synthetic camera, detector, estimator and controller. Recorded DRACO images are evaluated
    separately and do not react to new commands.</p><img src="final_dashboard.png">
    <p><a href="summary.json">Results</a> | <a href="telemetry.csv">Telemetry</a> |
    <a href="config.json">Configuration</a></p></html>"""
    (out/"index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/visual_2d"))
    parser.add_argument("--show", action="store_true", help="Live OpenCV windows if GUI available")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--ballistic", action="store_true")
    parser.add_argument("--no-measurements", action="store_true")
    parser.add_argument("--reference-image", type=Path, help="Use detected body patches as appearance on modeled disks")
    parser.add_argument("--allow-opencv4", action="store_true", help="Explicit OpenCV 4 compatibility test")
    args = parser.parse_args()
    data = json.loads(args.config.read_text()) if args.config else {}
    if args.ballistic: data["control"] = False
    if args.no_measurements: data["measurements"] = False
    if args.allow_opencv4: data["require_opencv5"] = False
    if args.reference_image: data["reference_image"] = str(args.reference_image.resolve())
    cfg = PlanarConfig(**data); cfg.validate()
    cv = require_opencv()
    if cfg.require_opencv5 and int(cv.__version__.split(".")[0]) != 5:
        parser.error(f"OpenCV 5 required, loaded {cv.__version__}; install .[vision]")
    args.out.mkdir(parents=True, exist_ok=True)
    movie = Movie(args.out, cfg.fps, not args.no_video); start = time.monotonic()
    try:
        if not args.no_overview:
            binary = Binary(period_s=cfg.period_s, primary_spin_period_s=cfg.spin_period_s,
                            separation_m=cfg.separation_m, phase_rad=cfg.phase_rad)
            for index in range(cfg.overview_frames):
                frame = orbit_frame(cfg, binary, index); movie.append(frame)
                if index == cfg.overview_frames//2:
                    cv.imwrite(str(args.out/"orbit_overview.png"), frame)
        def callback(*inputs):
            frame = dashboard(*inputs); movie.append(frame); row = inputs[2]
            if int(row["elapsed_s"]) in (0, 210, 225):
                cv.imwrite(str(args.out/f"frame_{int(row['elapsed_s']):04d}.png"), frame)
            if args.show:
                cv.imshow("OpenDART 2D", frame)
                if cv.waitKey(1)&255 == 27: raise KeyboardInterrupt
            return frame
        result = simulate_planar(cfg, callback)
        if result["final_dashboard"] is not None:
            final = result["final_dashboard"].copy()
            cv.rectangle(final, (720, 80), (1232, 147), BG, -1)
            text(final, "RESULT: "+result["summary"]["status"], (735, 108), .65, GOLD, 2)
            text(final, f"event t={result['summary']['elapsed_s']:.3f}s; camera t="
                 f"{result['telemetry'][-1]['elapsed_s']:.1f}s", (735, 136), .43)
            result["final_dashboard"] = final
            for _ in range(int(2*cfg.fps)): movie.append(final)
    finally:
        movie.close()
        if args.show: cv.destroyAllWindows()
    save_outputs(result, args.out, movie, time.monotonic()-start)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
