"""Person crop quality gate (implements the `CropSelector` port for people).

A ReID vector is only as good as the crop it came from. A 40 px silhouette, a
person cut off by the frame edge, or a box around two people all embed fine
and match the wrong people. This decides which person crops may become
gallery exemplars, and scores the rest 0..1 so the best ones win.

The numbers are a starting point (docs/VIP-9-PERSON-PLAN.md §3) to calibrate
on project footage -- not measured truths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.models import CropSample, ObjectType, TrackObservation


@dataclass(frozen=True)
class PersonCropRules:
    min_height: int = 96
    """Source pixels. Below ~90 px a person is a silhouette (semantic-va's limit)."""
    min_aspect: float = 0.5
    max_aspect: float = 5.0
    """Height / width. Outside this: several people in one box, or a sliver."""
    upright_aspect: float = 1.5
    posture_penalty: float = 0.6
    """Below `upright_aspect` the person is probably sitting or bending: still a
    person, but ReID models are trained on people standing, so such a crop is
    kept with lower quality and a standing view of the same track wins.
    (First MEVA test, 2026-10-19: every "odd shape" rejection under the old
    1.5 minimum was a clearly visible seated person, h/w 0.77-0.92.)"""
    min_confidence: float = 0.5
    """Weaker boxes may keep a track alive; they aren't gallery material."""
    edge_margin: float = 0.01
    edge_penalty: float = 0.5
    """A box touching the frame border is probably cut off: its quality halves."""
    full_height: int = 256
    """Height that earns full marks for size (OSNet's input height)."""
    min_sharpness: float = 0.0
    """Variance of the Laplacian; 0 disables the check until a floor is measured."""


@dataclass(frozen=True)
class CropVerdict:
    quality: float
    """0..1; 0 when rejected."""
    reason: str | None
    """Why it was rejected, or None."""
    box: tuple[int, int, int, int]
    """The crop in pixels after clamping to the frame."""


def judge_person_crop(image: np.ndarray, bbox: tuple[float, float, float, float],
                      confidence: float, rules: PersonCropRules = PersonCropRules()) -> CropVerdict:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box = (max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))
    cw, ch = box[2] - box[0], box[3] - box[1]
    if cw <= 0 or ch <= 0:
        return CropVerdict(0.0, "empty box", box)
    if confidence < rules.min_confidence:
        return CropVerdict(0.0, "low confidence", box)
    if ch < rules.min_height:
        return CropVerdict(0.0, "too small", box)
    if not rules.min_aspect <= ch / cw <= rules.max_aspect:
        return CropVerdict(0.0, "odd shape", box)
    if rules.min_sharpness > 0 and sharpness(image[box[1]:box[3], box[0]:box[2]]) < rules.min_sharpness:
        return CropVerdict(0.0, "blurry", box)

    quality = min(1.0, ch / rules.full_height) * confidence
    if ch / cw < rules.upright_aspect:
        quality *= rules.posture_penalty
    m = rules.edge_margin
    if x1 < m * w or y1 < m * h or x2 > (1 - m) * w or y2 > (1 - m) * h:
        quality *= rules.edge_penalty
    return CropVerdict(round(quality, 4), None, box)


def sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian: low for blurred crops. NumPy only (no OpenCV)."""
    gray = crop.astype(np.float32).mean(axis=2)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


class PersonCropSelector:
    """Per observation: a person crop with its quality, or None.

    Choosing the best few crops over a WHOLE track needs to see the track end;
    that arrives with the tracker (milestone M2)."""

    def __init__(self, rules: PersonCropRules = PersonCropRules()) -> None:
        self.rules = rules
        self.rejected: dict[str, int] = {}

    def select(self, observation: TrackObservation) -> CropSample | None:
        if observation.object_type is not ObjectType.PERSON:
            return None
        verdict = judge_person_crop(observation.image, observation.bbox,
                                    observation.confidence, self.rules)
        if verdict.reason is not None:
            self.rejected[verdict.reason] = self.rejected.get(verdict.reason, 0) + 1
            return None
        x1, y1, x2, y2 = verdict.box
        return CropSample(observation.track_id, ObjectType.PERSON,
                          observation.image[y1:y2, x1:x2].copy(), verdict.quality)
