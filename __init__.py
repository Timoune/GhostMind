"""
GhostMind
=========

Autonomous cognitive reasoning engine for Mini Von.

Public API (recommended usage):

    from ghostmind.core.runtime import Runtime
    from ghostmind.cognition.pipeline import CognitionPipeline
    from ghostmind.cognition.cognition_types import (
        Uncertainty, Assumption, IntentAnalysis, ...
    )

When used inside the MiniVon monorepo, you can also do:

    from MiniVon.GhostMind.core.runtime import Runtime
"""

from .core.runtime import Runtime
from .cognition.pipeline import CognitionPipeline

__version__ = "4.5.0"
__all__ = [
    "Runtime",
    "CognitionPipeline",
    "__version__",
]
