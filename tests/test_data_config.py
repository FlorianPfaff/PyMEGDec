import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pymegdec.data_config import DATA_DIR_ENV_VAR, resolve_data_folder


class TestResolveDataFolder(unittest.TestCase):
    def test_explicit_data_folder_takes_precedence(self):
        with (
            tempfile.TemporaryDirectory() as explicit_dir,
            tempfile.TemporaryDirectory() as env_dir,
        ):
            with patch.dict(os.environ, {DATA_DIR_ENV_VAR: env_dir}):
                self.assertEqual(
                    resolve_data_folder(explicit_dir),
                    str(Path(explicit_dir).resolve()),
                )

    def test_env_var_is_used_without_explicit_argument(self):
        with tempfile.TemporaryDirectory() as env_dir:
            with patch.dict(os.environ, {DATA_DIR_ENV_VAR: env_dir}):
                with patch("pymegdec.data_config._read_local_data_dir_file", return_value=None):
                    self.assertEqual(
                        resolve_data_folder(),
                        str(Path(env_dir).resolve()),
                    )

    def test_local_config_file_resolves_relative_to_config_directory(self):
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            (root / ".pymegdec-data-dir").write_text("data\n", encoding="utf-8")

            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(resolve_data_folder(), str(data_dir.resolve()))
            finally:
                os.chdir(previous_cwd)

    def test_required_files_are_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_data_folder(
                    temp_dir,
                    required=True,
                    required_files=["Part2Data.mat"],
                )


if __name__ == "__main__":
    unittest.main()
