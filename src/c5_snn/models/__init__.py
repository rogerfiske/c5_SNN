"""Model definitions and registry for the c5_SNN pipeline."""

from c5_snn.models.base import MODEL_REGISTRY, BaseModel, get_model
from c5_snn.models.baselines import FrequencyBaseline, GRUBaseline
from c5_snn.models.encoding import SpikeEncoder
from c5_snn.models.snn_models import SpikingCNN1D, SpikingMLP

__all__ = [
    "BaseModel",
    "FrequencyBaseline",
    "GRUBaseline",
    "MODEL_REGISTRY",
    "SpikeEncoder",
    "SpikingCNN1D",
    "SpikingMLP",
    "get_model",
]
