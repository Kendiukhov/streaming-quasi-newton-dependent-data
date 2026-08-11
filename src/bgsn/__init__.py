"""Blocked-Gapped Streaming Newton (BGSN): streaming inversion-free quasi-Newton
estimation and inference for temporally dependent data streams."""
from . import models, streams, estimators, _core  # noqa: F401

__all__ = ["models", "streams", "estimators", "_core"]
__version__ = "1.0.0"
