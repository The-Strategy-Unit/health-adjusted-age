"""Public API"""

from .config import DEFAULT_CONFIG, ModelConfig
from .fitting import fit_all_metalogs

__all__ = [
    "fit_all_metalogs",
    "ModelConfig",
    "DEFAULT_CONFIG",
]
