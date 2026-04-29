# PyMEGDec

Utilities for MEG decoding experiments, including model transfer between experiment
conditions and cross-validation on a single dataset.

## Repository layout

```text
src/pymegdec/              Package source code
  alpha_signal.py          Alpha-band filtering and phase extraction
  alpha_visualization.py   Alpha signal and phase-shift plotting helpers
  classifiers.py           Classifier factories and PyTorch Lightning model
  preprocessing.py         Filtering, downsampling, window extraction, PCA
  model_transfer.py        Train-on-experiment / validate-on-cue evaluation
  cross_validation.py      Single-dataset cross-validation routine
tests/                     Unit and data-dependent unittest suites
.github/workflows/         CI jobs for unit and data-dependent test subsets
```

Top-level `cross_validation.py`, `evaluate_model_transfer.py`,
`extract_alpha_signal.py`, and `show_bandpass_signal_and_shifts.py` are
compatibility wrappers for existing imports and direct script usage.

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional classifier backends when needed:

```bash
python -m pip install -e ".[all]"
```

## Data directory

Participant data is configured at runtime so private or machine-specific paths
do not need to be committed. Data files are expected to be named like
`Part2Data.mat` and `Part2CueData.mat`.

Resolution order:

1. Pass `--data-dir` to a CLI command, or pass `data_folder` to the Python API.
2. Set the `PYMEGDEC_DATA_DIR` environment variable.
3. Create a local `.pymegdec-data-dir` file containing one path. The resolver
   searches the current directory, its parents, and the project root.
4. Fall back to the current working directory for backwards compatibility.

`.pymegdec-data-dir` is ignored by git and can contain a path relative to the
file location.

On PowerShell:

```powershell
$env:PYMEGDEC_DATA_DIR = "C:\path\to\data"
python -m unittest
```

## CLI usage

```bash
pymegdec-cross-validate --data-dir "/path/to/MEG-Data" --participant 2
pymegdec-transfer --data-dir "/path/to/MEG-Data" --participant 2 --null-window-center nan
```

The grouped command exposes the same workflows:

```bash
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
```

## Examples

```python
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.cross_validation import cross_validate_single_dataset

transfer_accuracy = evaluate_model_transfer("/path/to/MEG-Data", 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset("/path/to/MEG-Data", 2, classifier="multiclass-svm")
```

If `PYMEGDEC_DATA_DIR` or `.pymegdec-data-dir` is configured, the first argument
can be `None`:

```python
transfer_accuracy = evaluate_model_transfer(None, 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(None, 2, classifier="multiclass-svm")
```

## Tests

The default suite includes fast tests that run without private MEG files.
Data-dependent accuracy checks are skipped when the data directory cannot be
resolved.

```bash
python -m unittest discover -v
```

To run the data-dependent integration tests, point `PYMEGDEC_DATA_DIR` at a
directory containing `Part2Data.mat` and `Part2CueData.mat` before running the
same command.
