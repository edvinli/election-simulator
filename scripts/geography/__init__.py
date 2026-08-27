"""Geographic Projection package for Swedish Riksdag simulator."""

from .config import MODEL_PARTIES_9, REST_MANDATE_LABEL
from .evaluate import evaluate_projection_pair, run_all_historical_evaluations
from .projection import ProjectionResult, project_constituency_votes
from .raking import IPFResult, iterative_proportional_fitting

__all__ = [
    "MODEL_PARTIES_9",
    "REST_MANDATE_LABEL",
    "IPFResult",
    "iterative_proportional_fitting",
    "ProjectionResult",
    "project_constituency_votes",
    "evaluate_projection_pair",
    "run_all_historical_evaluations",
]
