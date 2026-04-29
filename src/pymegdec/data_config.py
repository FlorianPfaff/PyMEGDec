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


def _read_local_data_dir_file() -> tuple[str, Path] | None:
    search_dirs = [Path.cwd(), *Path.cwd().parents]

    package_path = Path(__file__).resolve()
    if len(package_path.parents) > 2 and package_path.parents[1].name == "src":
        search_dirs.append(package_path.parents[2])

    seen_dirs = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)

        config_path = directory / LOCAL_DATA_DIR_FILE
        if not config_path.exists():
            continue

        for line in config_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value, directory

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
    required_files: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Resolve the directory containing participant ``.mat`` files.

    Resolution order is explicit argument, ``PYMEGDEC_DATA_DIR``, local
    ``.pymegdec-data-dir`` config file, then the current working directory for
    backwards compatibility.
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
        local_config = _read_local_data_dir_file()
        if local_config:
            raw_data_folder, relative_to = local_config
            source = LOCAL_DATA_DIR_FILE
        else:
            raw_data_folder = "."

    path = _resolve_path(raw_data_folder, relative_to=relative_to)
    missing_files = [str(name) for name in required_files if not (path / name).exists()]

    if required and (not path.exists() or missing_files):
        detail = (
            f"Missing required data files: {', '.join(missing_files)}."
            if missing_files
            else "Data directory does not exist."
        )
        raise FileNotFoundError(
            f"{detail} Set {DATA_DIR_ENV_VAR}, pass data_folder, or create an ignored "
            f"{LOCAL_DATA_DIR_FILE} file. Resolved {source} to: {path}"
        )

    return str(path)
