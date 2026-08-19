"""V2 component-consensus backend and stage APIs."""

from trackrefinery.component_consensus.aggregation import (
    aggregate_geometry_components,
)
from trackrefinery.component_consensus.components import select_object_components
from trackrefinery.component_consensus.pose_graph import (
    POSE_GRAPH_EXPERIMENT_CONTRACT,
    POSE_GRAPH_VARIANTS,
    PoseGraphAggregationRun,
    PoseGraphEdgeTrace,
    PoseGraphExperimentSettings,
    PoseGraphExperimentTrace,
    aggregate_geometry_components_pose_graph,
)
from trackrefinery.component_consensus.refiner import ComponentConsensusRefiner
from trackrefinery.component_consensus.settings import (
    COMPONENT_CONSENSUS_ALGORITHM_VERSION,
    COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION,
    ComponentConsensusSettings,
)

__all__ = [
    "COMPONENT_CONSENSUS_ALGORITHM_VERSION",
    "COMPONENT_CONSENSUS_CONFIG_SCHEMA_VERSION",
    "POSE_GRAPH_EXPERIMENT_CONTRACT",
    "POSE_GRAPH_VARIANTS",
    "ComponentConsensusRefiner",
    "ComponentConsensusSettings",
    "PoseGraphAggregationRun",
    "PoseGraphEdgeTrace",
    "PoseGraphExperimentSettings",
    "PoseGraphExperimentTrace",
    "aggregate_geometry_components",
    "aggregate_geometry_components_pose_graph",
    "select_object_components",
]
