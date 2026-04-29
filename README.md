# PyMEGDec

Utilities for evaluating MEG decoding models with cross-validation and
model-transfer experiments.

## Installation

Install the package in editable mode from the repository root:

```powershell
python -m pip install -e .
```

Install the optional machine-learning extras when using `xgboost` or
`pytorch-mlp` classifiers:

```powershell
python -m pip install -e ".[ml]"
```

## Data Directory

Participant data is configured at runtime so machine-specific or private data
paths do not need to be committed to the repository.

Resolution order:

1. Pass `--data-dir` to a CLI command, or pass `data_folder` to the Python API.
2. Set the `PYMEGDEC_DATA_DIR` environment variable.
3. Create a local `.pymegdec-data-dir` file in the working directory or project root.
4. Fall back to the current working directory for backwards compatibility.

`.pymegdec-data-dir` is ignored by git and should contain one local path.

## CLI Usage

```powershell
pymegdec-cross-validate --data-dir "C:\path\to\MEG-Data" --participant 2
pymegdec-transfer --data-dir "C:\path\to\MEG-Data" --participant 2 --null-window-center nan
```

The grouped command exposes the same workflows:

```powershell
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
```

## Python API

```python
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.evaluate_model_transfer import evaluate_model_transfer

accuracy = cross_validate_single_dataset(data_folder="C:/path/to/MEG-Data", participant_id=2)
transfer_accuracy = evaluate_model_transfer(data_folder="C:/path/to/MEG-Data", parts=2)
```

## Tests

The default suite includes synthetic-data tests that run without private MEG
files. Accuracy checks that use real participant `.mat` files are skipped when
the data directory cannot be resolved.

```powershell
python -m unittest
```

To run the data-dependent integration tests, point `PYMEGDEC_DATA_DIR` at a
directory containing `Part2Data.mat` and `Part2CueData.mat` before running the
same command.
