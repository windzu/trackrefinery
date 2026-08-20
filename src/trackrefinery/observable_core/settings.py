"""Versioned settings for deterministic observable-core selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from math import isfinite

from trackrefinery.component_consensus.settings import ComponentConsensusSettings

OBSERVABLE_CORE_ALGORITHM_VERSION = "observable-core-refiner-v1.0.0"
OBSERVABLE_CORE_CONFIG_SCHEMA_VERSION = "trackrefinery-observable-core-settings-v1"


@dataclass(frozen=True, slots=True)
class ObservableCoreSettings:
    """Internal component evidence and temporal-connectivity policy."""

    component: ComponentConsensusSettings = field(
        default_factory=ComponentConsensusSettings
    )
    maximum_timestamp_gap_factor: float = 2.5

    def __post_init__(self) -> None:
        factor = float(self.maximum_timestamp_gap_factor)
        if not isfinite(factor) or factor < 1.0:
            raise ValueError(
                "maximum_timestamp_gap_factor must be finite and at least one"
            )
        object.__setattr__(self, "maximum_timestamp_gap_factor", factor)

    @property
    def minimum_core_frames(self) -> int:
        return self.component.track_minimum_geometry_frames

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OBSERVABLE_CORE_CONFIG_SCHEMA_VERSION,
            "maximum_timestamp_gap_factor": self.maximum_timestamp_gap_factor,
            "minimum_core_frames": self.minimum_core_frames,
            "component": self.component.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
