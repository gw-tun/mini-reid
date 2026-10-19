from __future__ import annotations

from collections.abc import Sequence

from ..domain.models import Frame, FollowSession, MatchCandidate
from ..domain.ports import CropSelector, Detector, Tracker
from .live_follow import LiveFollowService


class LiveCameraService:
    """Runs one camera's detector/tracker loop and hands candidates to Live Follow."""

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        crop_selector: CropSelector,
        follow: LiveFollowService,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._crop_selector = crop_selector
        self._follow = follow

    def process_frame(self, frame: Frame, session: FollowSession) -> Sequence[MatchCandidate]:
        matches: list[MatchCandidate] = []
        detections = self._detector.detect(frame)
        for observation in self._tracker.update(frame, detections):
            crop = self._crop_selector.select(observation)
            if crop is None:
                continue
            matches.extend(self._follow.observe(session, observation, crop))
        return tuple(matches)
