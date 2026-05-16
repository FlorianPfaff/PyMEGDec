"""Public compatibility facade for cross-subject stimulus decoding.

The implementation lives in :mod:`pymegdec._stimulus_cross_subject_core`.
This module is kept as the stable public import path used by PyMEGDec's
CLI, tests, and downstream modules.
"""

from __future__ import annotations

import sys

from pymegdec import _stimulus_cross_subject_core as _core

# Make imports of ``pymegdec.stimulus_cross_subject`` resolve to the core module
# object.  This keeps private helper monkey-patches and existing direct imports
# operating on the implementation module rather than on a shallow re-export copy.
sys.modules[__name__] = _core
globals().update(_core.__dict__)
