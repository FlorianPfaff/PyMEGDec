"""Helpers for running repository scripts without installing the package."""

import sys
from pathlib import Path


def add_src_to_path(script_file):
    """Add the repository ``src`` directory for direct script execution."""

    src_dir = Path(script_file).resolve().parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
