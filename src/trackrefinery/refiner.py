"""Stable algorithm interface; concrete refinement is intentionally absent."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trackrefinery.contracts import (
    InsufficientEvidence,
    RefinementCase,
    RefinementOutcome,
    RefinementSuccess,
)


class TrackRefiner(ABC):
    """Base class for one-instance, whole-track refinement implementations."""

    def refine(self, case: RefinementCase) -> RefinementOutcome:
        """Run a backend and enforce the public result invariants."""

        outcome = self._refine(case)
        validate_outcome(case, outcome)
        return outcome

    @abstractmethod
    def _refine(self, case: RefinementCase) -> RefinementOutcome:
        """Implement canonical-size and per-frame-pose estimation."""


def validate_outcome(case: RefinementCase, outcome: RefinementOutcome) -> None:
    """Validate that an outcome belongs to and completely covers its input."""

    if outcome.track_id != case.track.track_id:
        raise ValueError("outcome track_id does not match the input track")
    if isinstance(outcome, RefinementSuccess):
        expected = [item.frame_id for item in case.track.observations]
        actual = [item.frame_id for item in outcome.frame_poses]
        if actual != expected:
            raise ValueError("a successful outcome must preserve every input frame")
    elif not isinstance(outcome, InsufficientEvidence):
        raise TypeError("backend returned an unsupported refinement outcome")
