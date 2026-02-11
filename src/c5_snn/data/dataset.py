"""PyTorch Dataset and DataLoader factory for CA5 windowed tensors."""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

from c5_snn.data.splits import SplitInfo

logger = logging.getLogger(__name__)


class CA5Dataset(Dataset):
    """PyTorch Dataset wrapping a slice of windowed tensors.

    Args:
        X: Input tensor of shape (N, W, 39).
        y: Target tensor of shape (N, 39).
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    split_info: SplitInfo,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> dict[str, DataLoader]:
    """Create DataLoaders for each split.

    All DataLoaders use shuffle=False to preserve chronological order.

    Args:
        split_info: SplitInfo with index ranges per split.
        X: Full input tensor of shape (N, W, 39).
        y: Full target tensor of shape (N, 39).
        batch_size: Batch size for all DataLoaders.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    loaders: dict[str, DataLoader] = {}

    for split_name in ("train", "val", "test"):
        start, end = split_info.indices[split_name]
        dataset = CA5Dataset(X[start:end], y[start:end])
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        logger.info(
            "Created %s DataLoader: %d samples, batch_size=%d",
            split_name,
            len(dataset),
            batch_size,
        )

    return loaders
