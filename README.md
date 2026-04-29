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

The test suite expects MATLAB data files named like `Part2Data.mat` and
`Part2CueData.mat` in the working directory. CI downloads these files from
repository secrets before running the tests.

## Examples

```python
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.cross_validation import cross_validate_single_dataset

transfer_accuracy = evaluate_model_transfer(".", 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(".", 2, classifier="multiclass-svm")
```
