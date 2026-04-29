# PyMEGDec

Utilities for MEG decoding experiments, including model transfer between experiment
conditions and cross-validation on a single dataset.

## Repository layout

```text
src/pymegdec/              Package source code
  alpha_signal.py          Alpha-band filtering and phase extraction
  alpha_metrics.py         Per-trial alpha power and phase-gradient export
  alpha_visualization.py   Alpha signal and phase-shift plotting helpers
  classifiers.py           Classifier factories and PyTorch Lightning model
  preprocessing.py         Filtering, downsampling, window extraction, PCA
  model_transfer.py        Train-on-experiment / validate-on-cue evaluation
  cross_validation.py      Single-dataset cross-validation routine
tests/                     Unit and data-dependent unittest suites
.github/workflows/         CI jobs for unit and data-dependent test subsets
```

Top-level `cross_validation.py`, `evaluate_model_transfer.py`,
`extract_alpha_signal.py`, `show_bandpass_signal_and_shifts.py`, and
`export_alpha_metrics.py` are compatibility wrappers for existing imports and
direct script usage.

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

The data directory is configured at runtime so private or machine-specific paths
do not need to be committed. Data files are expected to be named like
`Part2Data.mat` and `Part2CueData.mat`.

Resolution order:

1. Pass a data directory to the Python API or command-line wrapper.
2. Set the `PYMEGDEC_DATA_DIR` environment variable.
3. Create a local `.pymegdec-data-dir` file containing one path. This file is
   ignored by git.
4. Fall back to the current working directory for backwards compatibility.

On PowerShell:

```powershell
$env:PYMEGDEC_DATA_DIR = "C:\path\to\data"
python -m unittest
```

Or pass the directory explicitly:

```powershell
python cross_validation.py --data-dir "C:\path\to\data" --participant 2
python evaluate_model_transfer.py --data-dir "C:\path\to\data" --participant 2
```

## Examples

```python
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.cross_validation import cross_validate_single_dataset

transfer_accuracy = evaluate_model_transfer(".", 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(".", 2, classifier="multiclass-svm")
```

If `PYMEGDEC_DATA_DIR` or `.pymegdec-data-dir` is configured, the first argument
can be `None`:

```python
transfer_accuracy = evaluate_model_transfer(None, 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(None, 2, classifier="multiclass-svm")
```

## Exploratory alpha metrics

Prestimulus alpha metrics can be exported per trial for downstream plotting or
statistics. By default the exporter uses the `MLO*`, `MRO*`, and `MZO*`
occipital CTF channels and the `-0.4` to `-0.05 s` window before stimulus
onset.

```powershell
python export_alpha_metrics.py --participant 2 --output outputs\part2_alpha_metrics.csv
python export_alpha_metrics.py --participant 2 --cue --output outputs\part2_cue_alpha_metrics.csv
```

The exported rows include alpha power, phase concentration, planar phase-fit
quality, spatial phase frequency, estimated propagation speed, and dominant
phase-gradient direction on a projected sensor plane. The `outputs/` directory
is ignored by git.
