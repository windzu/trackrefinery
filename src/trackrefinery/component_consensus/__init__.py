"""V2 component-consensus backend and stage APIs."""

from trackrefinery.component_consensus.aggregation import (
    aggregate_geometry_components,
)
from trackrefinery.component_consensus.components import select_object_components
from trackrefinery.component_consensus.refiner import ComponentConsensusRefiner
from trackrefinery.component_consensus.settings import (
    COMPONENT_CONSENSUS_ALGORITHM_VERSION,
    COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
    ComponentConsensusSettings,
)

__all__ = [
    "COMPONENT_CONSENSUS_ALGORITHM_VERSION",
    "COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION",
    "ComponentConsensusRefiner",
    "ComponentConsensusSettings",
    "aggregate_geometry_components",
    "select_object_components",
]
