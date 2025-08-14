"""
WavePose: WiFi-based Human Pose Estimation

A deep learning framework that uses WiFi Channel State Information (CSI)
to reconstruct dense 2D/3D human pose maps without cameras or privacy concerns.
"""

__version__ = "1.0.0"
__author__ = "WavePose Team"
__email__ = "contact@wavepose.ai"

from . import data
from . import models
from . import utils
from . import training
from . import inference

__all__ = [
    "data",
    "models", 
    "utils",
    "training",
    "inference"
]
