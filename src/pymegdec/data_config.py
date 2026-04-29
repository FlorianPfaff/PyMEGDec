"""Helpers for locating private PyMEGDec data files.

Data paths are intentionally resolved at runtime so machine-specific paths do
not need to be committed to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

DATA_DIR_ENV_VAR = "PYMEGDEC_DATA_DIR"
LOCAL_DATA_DIR_FILE = ".pymegdec-data-dir"


def _local_config_paths() -> list[Path]:
    package_root = Path(__file__).resolve().parents[2]
    return [
        Path.cwd() / LOCAL_DATA_DIR_FILE,
        package_root / LOCAL_DATA_DIR_FILE,
    ]


def _read_local_data_dir_file() -> tuple[str, Path] | None:
    for config_path in _local_config_paths():
        if not config_path.exists():
            continue

        for line in config_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value, config_path.parent

    return None


def _resolve_path(
    value: str | os.PathLike[str], *, relative_to: Path | None = None
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve()


def resolve_data_folder(
    data_folder: str | os.PathLike[str] | None = None,
    *,
    required: bool = False,
    required_files: Iterable[str] = (),
) -> str:
    """Resolve the directory containing participant ``.mat`` files.

    Resolution order is:
    1. explicit ``data_folder`` argument,
    2. ``PYMEGDEC_DATA_DIR`` environment variable,
    3. local ``.pymegdec-data-dir`` file in the working directory or project root,
    4. current working directory, preserving the historical default.
    """

    source = "current working directory"
    relative_to: Path | None = None

    if data_folder:
        raw_data_folder = data_folder
        source = "function argument"
    elif os.environ.get(DATA_DIR_ENV_VAR):
        raw_data_folder = os.environ[DATA_DIR_ENV_VAR]
        source = DATA_DIR_ENV_VAR
    else:
        local_data_folder = _read_local_data_dir_file()
        if local_data_folder is not None:
            raw_data_folder, relative_to = local_data_folder
            source = LOCAL_DATA_DIR_FILE
        else:
            raw_data_folder = "."

    path = _resolve_path(raw_data_folder, relative_to=relative_to)
    missing_files = [name for name in required_files if not (path / name).exists()]

    if required and (not path.exists() or missing_files):
        detail = (
            f"Missing required data files: {', '.join(missing_files)}."
            if missing_files
            else "Data directory does not exist."
        )
        raise FileNotFoundError(
            f"{detail} Set {DATA_DIR_ENV_VAR}, pass data_folder, or create "
            f"an ignored {LOCAL_DATA_DIR_FILE} file in the repository root. "
            f"Resolved {source} to: {path}"
        )

    return str(path)
