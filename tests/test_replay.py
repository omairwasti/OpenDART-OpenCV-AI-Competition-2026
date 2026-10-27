import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
from opendart.replay import observation_times,resolve_images,spacecraft_clock_ticks


def test_sclk_fraction_timing_preserves_ticks_and_duplicate_capture():
    files = [Path("dart_0401929789_00001_01_iof.png"),
             Path("dart_0401929788_65535_02_iof.png"),
             Path("dart_0401929788_65535_01_iof.png")]
    ordered,times,relative,duplicates,metadata = observation_times(files)
    assert ordered[0].name.endswith("65535_01_iof.png")
    assert np.array_equal(times,[0.,0.,2/65536.])
    assert np.array_equal(times,relative)
    assert duplicates == [False,True,False]
    assert metadata["clock_is_utc"] is False
    assert metadata["tracker_time_units"] == "seconds"
    assert spacecraft_clock_ticks("dart_0401929788_65536_01_iof.png") is None


def test_unknown_cadence_uses_frame_units_and_natural_sort(tmp_path):
    for name in ["frame10.png","frame2.png","frame1.png"]:
        (tmp_path/name).touch()
    ordered = resolve_images(str(tmp_path))
    assert [p.name for p in ordered] == ["frame1.png","frame2.png","frame10.png"]
    _,times,relative,_,metadata = observation_times(ordered)
    assert relative is None
    assert np.array_equal(times,[0.,1.,2.])
    assert metadata["tracker_time_units"] == "frames"


def test_explicit_csv_is_authoritative_and_requires_coverage(tmp_path):
    files = [tmp_path/"image2.png",tmp_path/"image1.png"]
    timestamps = tmp_path/"times.csv"
    timestamps.write_text("filename,timestamp_s\nimage1.png,10.0\nimage2.png,12.5\n")
    ordered,times,relative,_,metadata = observation_times(files,timestamps)
    assert ordered[0].name == "image1.png"
    assert np.array_equal(times,[0.,2.5])
    assert np.array_equal(relative,[0.,2.5])
    assert metadata["csv_first_timestamp_s"] == 10.
    timestamps.write_text("filename,timestamp_s\nimage1.png,10.0\n")
    with pytest.raises(ValueError,match="Missing or ambiguous"):
        observation_times(files,timestamps)
