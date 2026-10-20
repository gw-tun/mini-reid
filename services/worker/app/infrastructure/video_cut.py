"""Cut a few seconds out of a recording into a new H.264 MP4, at full resolution.

The same way semantic-va's src/core/playback.py makes its previews (decode,
re-encode with libx264, first frame at 0), minus its 960 px cap: ReID needs
the pixels -- the person crop gate wants people at least 96 px tall.
Re-encoding (not copying packets) also means any source container, AVI
included, comes out as an MP4 that starts exactly at `start`, not at the
keyframe before it.
"""

from __future__ import annotations

import pathlib
from fractions import Fraction


def cut_video(source: pathlib.Path, start: float, duration: float, out: pathlib.Path) -> tuple[int, int, int]:
    """`start`..`start+duration` seconds into `source` -> `out`.
    Returns (frames, width, height); 0 frames means that stretch isn't in the file."""
    import av

    out.parent.mkdir(parents=True, exist_ok=True)
    frames, last_pts = 0, -1
    with av.open(str(source)) as recording:
        stream = recording.streams.video[0]
        stream.thread_type = "AUTO"
        origin = float((stream.start_time or 0) * stream.time_base)
        recording.seek(int((origin + start) / stream.time_base), stream=stream, backward=True)
        with av.open(str(out), mode="w", options={"movflags": "+faststart"}) as output:
            encoder = output.add_stream("libx264", rate=stream.average_rate or 25)
            encoder.width = stream.width // 2 * 2
            encoder.height = stream.height // 2 * 2
            encoder.pix_fmt = "yuv420p"
            encoder.codec_context.time_base = Fraction(1, 90000)
            encoder.options = {"preset": "veryfast", "crf": "23", "bf": "0"}
            for frame in recording.decode(stream):
                if frame.pts is None:
                    continue
                at = float(frame.pts * frame.time_base) - origin
                if at < start - 1e-6:
                    continue
                if at >= start + duration - 1e-6:
                    break
                pts = round((at - start) * 90000)
                if pts <= last_pts:
                    continue
                frame = frame.reformat(width=encoder.width, height=encoder.height, format="yuv420p")
                frame.pict_type = av.video.frame.PictureType.NONE  # let x264 choose its own GOP
                frame.pts, frame.time_base, last_pts = pts, Fraction(1, 90000), pts
                for packet in encoder.encode(frame):
                    output.mux(packet)
                frames += 1
            for packet in encoder.encode():
                output.mux(packet)
        return frames, encoder.width, encoder.height
