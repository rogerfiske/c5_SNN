"""Device detection for CPU/CUDA/ROCm environments."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect and return the best available torch device.

    Returns torch.device("cuda") if CUDA is available (covers both
    NVIDIA CUDA and AMD ROCm via PyTorch's HIP backend), otherwise
    returns torch.device("cpu").

    Logs the selected device at INFO level and warns if ROCm environment
    variables are detected but CUDA is not available.

    Returns:
        The selected torch.device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info("Using device: %s (%s)", device, device_name)
    else:
        device = torch.device("cpu")
        logger.info("Using device: cpu")

        # Check for ROCm environment that isn't working
        hsa_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        if hsa_version is not None:
            logger.warning(
                "HSA_OVERRIDE_GFX_VERSION=%s is set but CUDA/ROCm is not available. "
                "Check your ROCm + PyTorch installation.",
                hsa_version,
            )
        elif os.path.exists("/opt/rocm"):
            logger.warning(
                "ROCm installation detected at /opt/rocm but CUDA is not available. "
                "For AMD RX 6600M, set HSA_OVERRIDE_GFX_VERSION=10.3.0 and install "
                "the ROCm PyTorch wheel.",
            )

    return device
