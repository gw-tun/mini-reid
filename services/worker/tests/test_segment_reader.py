import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader, describe_segment

UTC = ZoneInfo("UTC")


def write_video(path, seconds=3.0, fps=10, size=(160, 96)):
    import av

    w, h = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for n in range(int(seconds * fps)):
            img = np.full((h, w, 3), (n * 8) % 255, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_mediamtx_name_gives_camera_and_start():
    camera, start = describe_segment("/recordings/cam01/2026-10-19_08-30-00.mp4", UTC)
    assert camera == "cam01" and start == dt.datetime(2026, 10, 19, 8, 30, tzinfo=UTC)


def test_meva_name_gives_camera_and_start():
    camera, start = describe_segment(
        "/x/2018-03-13.15-50-03.15-55-03.school.G336.r13.avi", UTC)
    assert camera == "G336" and start == dt.datetime(2018, 3, 13, 15, 50, 3, tzinfo=UTC)


def test_frames_are_sampled_with_recording_times(tmp_path):
    clip = tmp_path / "cam07" / "2026-10-19_08-30-00.mp4"
    clip.parent.mkdir()
    write_video(clip, seconds=3.0, fps=10)
    frames = list(PyAVSegmentReader().frames(str(clip), sample_fps=2))
    assert len(frames) == 6
    assert {f.camera_key for f in frames} == {"cam07"}
    start = describe_segment(clip)[1]
    offsets = [(f.captured_at - start).total_seconds() for f in frames]
    assert offsets == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], abs=0.05)
    assert frames[0].image.shape == (96, 160, 3) and frames[0].image.dtype == np.uint8


def test_unnamed_file_needs_a_start_time(tmp_path):
    clip = tmp_path / "clip.mp4"
    write_video(clip, seconds=1.0)
    with pytest.raises(ValueError, match="can't tell when"):
        list(PyAVSegmentReader().frames(str(clip), 2))
    start = dt.datetime(2026, 1, 1, tzinfo=UTC)
    frames = list(PyAVSegmentReader(camera_key="test", started_at=start).frames(str(clip), 2))
    assert frames[0].camera_key == "test" and frames[0].captured_at == start


def test_a_stretch_gives_the_same_frames_as_a_whole_file_read(tmp_path):
    """Seeking to the middle must not shift the sampling grid: overlapping
    stretches of one recording have to sample the very same frames."""
    import numpy as np

    path = tmp_path / "cam01" / "2026-10-19_08-30-00.mp4"
    path.parent.mkdir()
    write_video(path, seconds=6.0)
    whole = {f.captured_at: f.image for f in PyAVSegmentReader().frames(str(path), 2)}
    part = list(PyAVSegmentReader().frames(str(path), 2, start_s=2.2, end_s=4.0))
    start = dt.datetime(2026, 10, 19, 8, 30, tzinfo=dt.timezone.utc)
    assert [f.captured_at - start for f in part] == [dt.timedelta(seconds=s) for s in (2.5, 3.0, 3.5, 4.0)]
    for f in part:
        assert np.array_equal(f.image, whole[f.captured_at])
