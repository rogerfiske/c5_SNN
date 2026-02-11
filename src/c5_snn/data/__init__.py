"""Data loading, validation, windowing, and splitting for the CA5 dataset."""

from c5_snn.data.dataset import CA5Dataset, get_dataloaders
from c5_snn.data.loader import load_csv
from c5_snn.data.splits import SplitInfo, create_splits, load_splits, save_splits
from c5_snn.data.validation import P_COLUMNS, ValidationReport, validate
from c5_snn.data.windowing import build_windows, save_tensors

__all__ = [
    "CA5Dataset",
    "P_COLUMNS",
    "SplitInfo",
    "ValidationReport",
    "build_windows",
    "create_splits",
    "get_dataloaders",
    "load_csv",
    "load_splits",
    "save_splits",
    "save_tensors",
    "validate",
]
