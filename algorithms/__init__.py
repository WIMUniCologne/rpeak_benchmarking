"""R-peak detection algorithms.

All algorithms are exposed as top-level attributes so callers can do:

    import algos as algorithm
    found = algorithm.zhai(data, samplerate)
"""

from __future__ import annotations

# Classical / signal-processing algorithms
from .zhai import zhai
from .xia import xia
from .arteagaFalconi import arteagaFalconi
from .nguyen import nguyen
from .pantompkins import pantompkins
from .xu import xu
from .shaik import shaik
from .hamilton import hamilton
from .park import park
from .kumari import kumari
from .elgendi import elgendi

# Deep learning algorithms
from .zahid import zahid
from .laitala import laitala
from .xiang import xiang
from .celik import celik
from .han import han_cnn
from .han import han_rnn


__all__ = [
    "arteagaFalconi",
    "celik",
    "elgendi",
    "hamilton",
    "han_cnn",
    "han_rnn",
    "kumari",
    "laitala",
    "nguyen",
    "pantompkins",
    "park",
    "shaik",
    "xia",
    "xiang",
    "xu",
    "zahid",
    "zhai",
]
