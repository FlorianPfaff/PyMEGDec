"""Public compatibility facade for cross-subject stimulus decoding."""

from __future__ import annotations

import sys

from pymegdec import _stimulus_cross_subject_core as _core

_impl = _core._impl
sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
