"""Choose which crops of a track get embedded (wraps a class's crop gate).

A track is many frames of one person. Embedding every frame wastes ReID time
on near-identical crops (spec §6.3: "do not create a new ReID embedding for
every raw frame; refresh every 3-5 sampled frames"). This selector:

1. passes each observation through the class's quality gate (e.g. person rules);
2. keeps the track's `keep` best crops, at least `min_gap` apart in time;
3. returns a crop for embedding only when it is among those best ones AND it is
   the track's first good crop or `refresh_every` observations have passed.

`end(track_id)` gives back the best crops once the track is over -- the
track's evidence (`tracks.best_crop_uri`) and its gallery.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..domain.models import CropSample, TrackObservation
from ..domain.ports import CropSelector


@dataclass
class _TrackCrops:
    best: list[tuple[CropSample, dt.datetime]] = field(default_factory=list)
    seen: int = 0
    last_emit: int | None = None


class TrackCropSelector:
    def __init__(self, gate: CropSelector, *, keep: int = 4, refresh_every: int = 4,
                 min_gap: dt.timedelta = dt.timedelta(seconds=0.5)) -> None:
        self.gate = gate
        self.keep = keep
        self.refresh_every = refresh_every
        self.min_gap = min_gap
        self._tracks: dict[str, _TrackCrops] = {}

    def select(self, observation: TrackObservation) -> CropSample | None:
        crop = self.gate.select(observation)
        if crop is None:
            return None
        state = self._tracks.setdefault(observation.track_id, _TrackCrops())
        state.seen += 1
        if not self._offer(state, crop, observation.observed_at):
            return None
        due = state.last_emit is None or state.seen - state.last_emit >= self.refresh_every
        if not due:
            return None
        state.last_emit = state.seen
        return crop

    def best(self, track_id: str) -> list[CropSample]:
        state = self._tracks.get(track_id)
        return [c for c, _ in sorted(state.best, key=lambda x: -x[0].quality)] if state else []

    def end(self, track_id: str) -> list[CropSample]:
        """The track's best crops, best first; forgets the track."""
        return [c for c, _ in self.end_with_times(track_id)]

    def end_with_times(self, track_id: str) -> list[tuple[CropSample, dt.datetime]]:
        """Like `end`, with when each crop was seen (to link it to its detection)."""
        state = self._tracks.pop(track_id, None)
        return sorted(state.best, key=lambda x: -x[0].quality) if state else []

    def _offer(self, state: _TrackCrops, crop: CropSample, at: dt.datetime) -> bool:
        """True when the crop entered the track's best set."""
        if any(abs(at - t) < self.min_gap for _, t in state.best):
            return False  # a near-duplicate of a crop already kept
        if len(state.best) < self.keep:
            state.best.append((crop, at))
            return True
        worst = min(range(len(state.best)), key=lambda i: state.best[i][0].quality)
        if crop.quality <= state.best[worst][0].quality:
            return False
        state.best[worst] = (crop, at)
        return True
