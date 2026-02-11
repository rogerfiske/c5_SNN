"""Model definitions and registry for the c5_SNN pipeline."""

from c5_snn.models.base import MODEL_REGISTRY, BaseModel, get_model
from c5_snn.models.baselines import FrequencyBaseline, GRUBaseline

__all__ = [
    "BaseModel",
    "FrequencyBaseline",
    "GRUBaseline",
    "MODEL_REGISTRY",
    "get_model",
]
