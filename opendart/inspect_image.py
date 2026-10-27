"""Inspect a local preview image without inferring missing mission geometry."""
import argparse
import json
from pathlib import Path

from .vision import AsteroidDetector, AsteroidTracker, annotate_candidates, require_opencv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/image_inspection"))
    args = parser.parse_args()
    cv = require_opencv()
    image = cv.imread(str(args.image), cv.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"Could not read image: {args.image}")
    detector = AsteroidDetector()
    candidates = detector.detect(image)
    tracker = AsteroidTracker()
    target = tracker.update(candidates, 0.)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "image": str(args.image.resolve()), "opencv_version": cv.__version__,
        "frame_size_px": list(image.shape[:2][::-1]),
        "status": tracker.status,
        "target_hypothesis": None if target is None else target.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "interpretation": "Optical blob measurements only; size-based body identity is a hypothesis. A single PNG does not provide time, calibrated range, attitude, or velocity.",
    }
    (args.out/"detections.json").write_text(json.dumps(result, indent=2)+"\n")
    for filename, pixels in [("annotated.png",annotate_candidates(image,candidates,target)),
                             ("segmentation.png",detector.last_binary)]:
        if not cv.imwrite(str(args.out/filename),pixels):
            raise OSError(f"Cannot write {filename}")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
