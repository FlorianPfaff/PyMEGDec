# PyMEGDec

Utilities for MEG decoding experiments, including model transfer between experiment
conditions and cross-validation on a single dataset.

## Repository layout

```text
src/pymegdec/              Package source code
  alpha_signal.py          Alpha-band filtering and phase extraction
  alpha_metrics.py         Per-trial alpha power and phase-gradient export
  alpha_movement.py        Sensor-level alpha movement trajectory export
  alpha_visualization.py   Alpha signal and phase-shift plotting helpers
  reaction_time_analysis.py Alpha/RT join and association summaries
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
direct script usage. `analyze_alpha_reaction_time.py` provides an exploratory
analysis command for alpha metrics and behavioral reaction times.
`analyze_alpha_movement.py` exports sensor-level alpha movement trajectories.

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

## Sensor-level alpha movement

The MAT files contain CTF sensor geometry in `data.grad.chanpos`, with positions
in millimeters. This supports sensor-array analyses of alpha topography, but not
source-localized brain movement. The movement exporter therefore tracks a
sensor-level proxy: for each trial and sampled time point, it filters the chosen
MEG channels to the alpha band, computes alpha power, and writes the
power-weighted centroid over the MEG sensor positions.

By default it uses all MEG channels matching `^M`, the `8-12 Hz` alpha band, and
a `-0.4` to `0.8 s` window around stimulus onset.

```powershell
python analyze_alpha_movement.py --participants 2 --trajectory-output outputs\part2_alpha_movement.csv --summary-output outputs\part2_alpha_movement_summary.csv
```

The trajectory CSV includes 3D CTF sensor centroids, projected 2D centroids,
stepwise speed, displacement from the first sampled time point, the peak-power
channel, and a spatial concentration score. Treat the trajectory as movement of
the measured alpha topography over sensors, not as anatomical source motion.

## Alpha and reaction time

The saved MEG `Part*Data.mat` files may not contain reaction times. The RT
analysis command therefore accepts an external behavioral CSV with
`participant`, `trial`, and `reaction_time` columns. If reaction times are stored
in a future MAT `trialinfo` column, pass `--trialinfo-rt-column` instead.

```powershell
python analyze_alpha_reaction_time.py --participants 2 --reaction-times behavior_rt.csv --joined-output outputs\part2_alpha_rt_trials.csv --summary-output outputs\part2_alpha_rt_summary.csv
```

The summary includes per-participant Pearson/regression rows and a pooled
within-participant row for each alpha metric. Phase-gradient direction is encoded
as sine and cosine components before analysis.
