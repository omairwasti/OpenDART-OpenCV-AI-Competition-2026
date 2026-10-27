"""Offline local DRACO preview playback: image tracking, without flight control.

PNG previews do not supply calibrated attitude or range. This tool measures
pixels and associates a secondary hypothesis; it never claims that the archive
images respond to a simulated spacecraft command. No archive is downloaded.
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from .vision import AsteroidDetector, AsteroidTracker, annotate_candidates, require_opencv


DEFAULT_IMAGE = str(Path(__file__).resolve().parents[1] / "assets" / "dart_0401929788_46871_01_iof.png")


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def resolve_images(specification):
    path = Path(specification).expanduser()
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    elif path.is_file():
        files = [path]
    else:
        files = [Path(p) for p in glob.glob(str(path), recursive=True)
                 if Path(p).is_file() and Path(p).suffix.lower() == ".png"]
    files = sorted(set(p.resolve() for p in files), key=natural_key)
    if not files:
        raise ValueError(f"No local PNG previews matched {specification!r}")
    return files


def spacecraft_clock_ticks(path):
    """Return integer DART filename clock ticks; do not convert SCLK to UTC.

    The second filename field is the 16-bit fractional spacecraft-clock field.
    Exact integer subtraction before conversion preserves subsecond spacing.
    Kernel-based clock correlation is required to obtain UTC or TDB.
    """
    match = re.match(r"^dart_(\d+)_(\d+)_", Path(path).name, re.IGNORECASE)
    if match is None:
        return None
    seconds, fraction = map(int, match.groups())
    if not 0 <= fraction < 65536:
        return None
    return seconds*65536+fraction


def observation_times(files, timestamps_csv=None):
    """Return ordered files, tracker times, physical relative times and metadata.

    Explicit CSV schema: filename,timestamp_s. Filenames may be absolute paths
    or unique basenames. Without timing information, tracker times are frame
    indices, labelled in frame units; the playback fps never fabricates cadence.
    """
    if timestamps_csv is not None:
        with Path(timestamps_csv).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"filename", "timestamp_s"}.issubset(reader.fieldnames or []):
                raise ValueError("Timestamp CSV requires filename,timestamp_s columns")
            rows = list(reader)
        lookup = {}
        for row in rows:
            name = row["filename"]
            if name in lookup:
                raise ValueError(f"Duplicate timestamp CSV filename: {name}")
            time = float(row["timestamp_s"])
            if not np.isfinite(time):
                raise ValueError("CSV timestamps must be finite seconds")
            lookup[name] = time
        names = [p.name for p in files]
        values = []
        for path in files:
            if str(path) in lookup:
                values.append(lookup[str(path)])
            elif path.name in lookup and names.count(path.name) == 1:
                values.append(lookup[path.name])
            else:
                raise ValueError(f"Missing or ambiguous timestamp for {path}")
        ordered = sorted(zip(files, values), key=lambda item:(item[1],natural_key(item[0])))
        files = [p for p,_ in ordered]
        original = np.array([value for _,value in ordered], dtype=float)
        relative = original-original[0]
        source = "explicit_csv_seconds"
        metadata = {"time_source":source,"tracker_time_units":"seconds",
                    "csv_first_timestamp_s":float(original[0]),"clock_is_utc":False}
    else:
        clocks = [spacecraft_clock_ticks(path) for path in files]
        if all(clock is not None for clock in clocks):
            ordered = sorted(zip(files,clocks), key=lambda item:(item[1],natural_key(item[0])))
            files = [p for p,_ in ordered]
            ticks = [clock for _,clock in ordered]
            relative = np.array([(tick-ticks[0])/65536. for tick in ticks])
            metadata = {"time_source":"filename_relative_sclk",
                        "tracker_time_units":"seconds", "sclk_ticks_per_second":65536,
                        "first_sclk_ticks":ticks[0],"clock_is_utc":False,
                        "clock_note":"Relative spacecraft-clock seconds; no UTC/TDB correlation applied."}
        else:
            relative = None
            metadata = {"time_source":"frame_index_cadence_unknown",
                        "tracker_time_units":"frames","clock_is_utc":False,
                        "clock_note":"No physical sampling cadence inferred. Playback fps is a display choice only."}
    tracker_times = np.arange(len(files),dtype=float) if relative is None else relative
    duplicates = [False]+[bool(tracker_times[i] == tracker_times[i-1]) for i in range(1,len(files))]
    metadata["duplicate_capture_frames"] = sum(duplicates)
    return files,tracker_times,relative,duplicates,metadata


def _video_frame(annotated, size, label, warning):
    cv = require_opencv()
    width,height = size
    scale = min(width/annotated.shape[1],height/annotated.shape[0])
    resized = cv.resize(annotated,(max(1,round(annotated.shape[1]*scale)),
                                 max(1,round(annotated.shape[0]*scale))))
    canvas = np.zeros((height+96,width,3),dtype=np.uint8)
    x,y = (width-resized.shape[1])//2,(height-resized.shape[0])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
    cv.putText(canvas,label,(8,height+25),cv.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1,cv.LINE_AA)
    cv.putText(canvas,"OFFLINE ARCHIVE: pixel tracking; no control feedback",(8,height+50),
               cv.FONT_HERSHEY_SIMPLEX,.45,(100,220,255),1,cv.LINE_AA)
    cv.putText(canvas,warning,(8,height+75),cv.FONT_HERSHEY_SIMPLEX,.4,(180,180,180),1,cv.LINE_AA)
    return canvas


def replay(images=DEFAULT_IMAGE, out="results/archive_replay", fps=20.,
           timestamps_csv=None, opencv5=True, max_frames=None):
    cv = require_opencv()
    if opencv5 and int(cv.__version__.split(".")[0]) < 5:
        raise RuntimeError(f"OpenCV 5 is required by default; found {cv.__version__}. Use --no-opencv5 only for compatibility checks.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Playback fps must be positive")
    files = resolve_images(images)
    files,times,relative,duplicates,timing = observation_times(files,timestamps_csv)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        files,times,duplicates = files[:max_frames],times[:max_frames],duplicates[:max_frames]
        relative = None if relative is None else relative[:max_frames]
    destination = Path(out)
    destination.mkdir(parents=True,exist_ok=True)
    detector,tracker = AsteroidDetector(),AsteroidTracker()
    fields = ["frame_index","filename","time_source","relative_time_s","tracker_time",
              "tracker_time_units","duplicate_capture","status","candidate_index",
              "selected_target","u_px","v_px","radius_px","area_px2",
              "cov_uu_px2","cov_uv_px2","cov_vv_px2","bbox_x","bbox_y",
              "bbox_width","bbox_height","touches_border","score"]
    writer = None
    acquired_frames = 0
    total_candidates = 0
    last_capture_time = None
    video_path = destination/"archive_replay.mp4"
    try:
        with (destination/"measurements.csv").open("w",newline="") as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=fields)
            csv_writer.writeheader()
            for frame,(path,time,is_duplicate) in enumerate(zip(files,times,duplicates)):
                image = cv.imread(str(path),cv.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"OpenCV cannot read {path}")
                candidates = detector.detect(image)
                # Repeated versions/previews at the same SCLK are displayed and
                # recorded but never counted as independent tracker updates.
                if is_duplicate and last_capture_time == time:
                    target = None
                    status = "DUPLICATE_CAPTURE_SKIPPED"
                else:
                    target = tracker.update(candidates,float(time))
                    status = tracker.status
                    last_capture_time = time
                acquired_frames += target is not None
                total_candidates += len(candidates)
                base = {"frame_index":frame,"filename":str(path),
                        "time_source":timing["time_source"],
                        "relative_time_s":"" if relative is None else f"{relative[frame]:.12f}",
                        "tracker_time":float(time),"tracker_time_units":timing["tracker_time_units"],
                        "duplicate_capture":is_duplicate,"status":status}
                if not candidates:
                    csv_writer.writerow(base)
                for index,candidate in enumerate(candidates):
                    x,y,w,h = candidate.bbox_xywh
                    row = dict(base,candidate_index=index,selected_target=candidate is target,
                               u_px=float(candidate.centroid_px[0]),v_px=float(candidate.centroid_px[1]),
                               radius_px=candidate.radius_px,area_px2=candidate.area_px2,
                               cov_uu_px2=float(candidate.covariance_px2[0,0]),
                               cov_uv_px2=float(candidate.covariance_px2[0,1]),
                               cov_vv_px2=float(candidate.covariance_px2[1,1]),
                               bbox_x=x,bbox_y=y,bbox_width=w,bbox_height=h,
                               touches_border=candidate.touches_border,score=candidate.score)
                    csv_writer.writerow(row)
                annotated = annotate_candidates(image,candidates,target)
                if writer is None:
                    # Keep original image coordinates in CSV; letterboxing is
                    # a display operation only. MPEG output has even dimensions.
                    size = (max(640,annotated.shape[1]+annotated.shape[1]%2),
                            annotated.shape[0]+annotated.shape[0]%2)
                    writer = cv.VideoWriter(str(video_path),cv.VideoWriter_fourcc(*"mp4v"),
                                            float(fps),(size[0],size[1]+96))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open the MP4 encoder (mp4v). Check video-codec support.")
                label = f"Frame {frame+1}/{len(files)}  {status}  "
                label += f"relative SCLK/CSV t={relative[frame]:.6f} s" if relative is not None else "physical cadence unknown"
                warning = "Coordinates: original image, zero-based u right / v down. Body identity is a hypothesis."
                output_frame = _video_frame(annotated,size,label,warning)
                writer.write(output_frame)
                if frame == 0 and not cv.imwrite(str(destination/"first_annotated.png"),output_frame):
                    raise OSError("Could not save first annotated replay frame")
    finally:
        if writer is not None:
            writer.release()
    timing["duplicate_capture_frames"] = sum(duplicates)
    summary = {"mode":"offline_archive_pixel_tracking","opencv_version":cv.__version__,
               "frame_count":len(files),"candidate_count":total_candidates,
               "associated_frames":acquired_frames,"playback_fps":float(fps),
               "playback_duration_s":len(files)/float(fps),
               "physical_duration_s":None if relative is None else float(relative[-1]-relative[0]),
               "timing":timing,"coordinate_convention":"Original images: zero-based u right, v down",
               "scope":"Image-only optical tracking. No spacecraft actuation, metric range, attitude, UTC or trajectory reconstructed from PNG previews.",
               "outputs":{"video":str(video_path),"measurements":str(destination/"measurements.csv"),
                          "first_annotated_frame":str(destination/"first_annotated.png")}}
    (destination/"replay_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images",default=DEFAULT_IMAGE,help="Local PNG, directory (recursive), or quoted glob")
    parser.add_argument("--out",default="results/archive_replay")
    parser.add_argument("--fps",type=float,default=20.,help="Video playback fps; never used as observation cadence")
    parser.add_argument("--timestamps",type=Path,help="Optional CSV filename,timestamp_s")
    parser.add_argument("--max-frames",type=int)
    parser.add_argument("--opencv5",action=argparse.BooleanOptionalAction,default=True)
    args = parser.parse_args()
    try:
        result = replay(args.images,args.out,args.fps,args.timestamps,args.opencv5,args.max_frames)
    except (ValueError,RuntimeError,ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
