"""Analyze and plot sensor-level alpha movement summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymegdec.alpha_metrics import write_alpha_metrics_csv

DEFAULT_PRE_WINDOW = (-0.4, -0.02)
DEFAULT_POST_WINDOW = (0.0, 0.4)
DEFAULT_ANALYSIS_METRICS = (
    "centroid_shift_mm",
    "projected_shift_mm",
    "post_minus_pre_speed_mm_per_s",
    "post_peak_speed_mm_per_s",
    "post_minus_pre_alpha_power",
    "post_minus_pre_spatial_concentration",
)


@dataclass(frozen=True)
class AlphaMovementAnalysisConfig:
    """Parameters for alpha movement summary analysis."""

    pre_window: tuple[float, float] = DEFAULT_PRE_WINDOW
    post_window: tuple[float, float] = DEFAULT_POST_WINDOW
    metrics: tuple[str, ...] = DEFAULT_ANALYSIS_METRICS
    plot_labels: tuple[str, ...] | None = None


def load_alpha_movement_rows(path):
    """Load alpha movement CSV rows."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows, output_path):
    """Write dictionaries to CSV."""

    write_alpha_metrics_csv(rows, output_path)


def _clean_id(value):
    if value is None:
        return ""
    return str(value)


def _to_float(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _window_rows(rows, window):
    start, stop = window
    if start >= stop:
        raise ValueError("Window start must be before stop.")
    return [row for row in rows if start <= _to_float(row.get("time_s")) <= stop]


def _mean(rows, key):
    values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _mean_vector(rows, keys):
    return np.asarray([_mean(rows, key) for key in keys], dtype=float)


def _peak_value_and_time(rows, value_key):
    best_value = -np.inf
    best_time = np.nan
    for row in rows:
        value = _to_float(row.get(value_key))
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_time = _to_float(row.get("time_s"))
    if not np.isfinite(best_value):
        return np.nan, np.nan
    return float(best_value), float(best_time)


def _participant_condition_key(row):
    return (
        _clean_id(row.get("participant")),
        _clean_id(row.get("dataset")),
        _clean_id(row.get("trial_label")),
    )


def _group_rows(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def _movement_effect_row(key, rows, config):
    participant, dataset, trial_label = key
    pre_rows = _window_rows(rows, config.pre_window)
    post_rows = _window_rows(rows, config.post_window)
    pre_centroid = _mean_vector(pre_rows, ("centroid_x_mm", "centroid_y_mm", "centroid_z_mm"))
    post_centroid = _mean_vector(post_rows, ("centroid_x_mm", "centroid_y_mm", "centroid_z_mm"))
    pre_projected = _mean_vector(pre_rows, ("projected_x_mm", "projected_y_mm"))
    post_projected = _mean_vector(post_rows, ("projected_x_mm", "projected_y_mm"))
    peak_speed, peak_speed_time = _peak_value_and_time(post_rows, "projected_speed_mm_per_s")

    return {
        "participant": participant,
        "dataset": dataset,
        "trial_label": trial_label,
        "pre_window_start": config.pre_window[0],
        "pre_window_stop": config.pre_window[1],
        "post_window_start": config.post_window[0],
        "post_window_stop": config.post_window[1],
        "n_pre_points": len(pre_rows),
        "n_post_points": len(post_rows),
        "pre_centroid_x_mm": pre_centroid[0],
        "pre_centroid_y_mm": pre_centroid[1],
        "pre_centroid_z_mm": pre_centroid[2],
        "post_centroid_x_mm": post_centroid[0],
        "post_centroid_y_mm": post_centroid[1],
        "post_centroid_z_mm": post_centroid[2],
        "centroid_shift_mm": float(np.linalg.norm(post_centroid - pre_centroid)),
        "pre_projected_x_mm": pre_projected[0],
        "pre_projected_y_mm": pre_projected[1],
        "post_projected_x_mm": post_projected[0],
        "post_projected_y_mm": post_projected[1],
        "projected_shift_mm": float(np.linalg.norm(post_projected - pre_projected)),
        "pre_speed_mm_per_s": _mean(pre_rows, "projected_speed_mm_per_s"),
        "post_speed_mm_per_s": _mean(post_rows, "projected_speed_mm_per_s"),
        "post_minus_pre_speed_mm_per_s": _mean(post_rows, "projected_speed_mm_per_s") - _mean(pre_rows, "projected_speed_mm_per_s"),
        "post_peak_speed_mm_per_s": peak_speed,
        "post_peak_speed_time_s": peak_speed_time,
        "pre_alpha_power": _mean(pre_rows, "mean_alpha_power"),
        "post_alpha_power": _mean(post_rows, "mean_alpha_power"),
        "post_minus_pre_alpha_power": _mean(post_rows, "mean_alpha_power") - _mean(pre_rows, "mean_alpha_power"),
        "pre_spatial_concentration": _mean(pre_rows, "spatial_concentration"),
        "post_spatial_concentration": _mean(post_rows, "spatial_concentration"),
        "post_minus_pre_spatial_concentration": _mean(post_rows, "spatial_concentration") - _mean(pre_rows, "spatial_concentration"),
    }


def analyze_alpha_movement_windows(rows, config=None):
    """Compute pre/post movement effects per participant and condition."""

    config = config or AlphaMovementAnalysisConfig()
    grouped = _group_rows(rows, _participant_condition_key)
    return [_movement_effect_row(key, group_rows, config) for key, group_rows in sorted(grouped.items())]


def _summary_stats(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), std, float(std / np.sqrt(values.size))


def summarize_alpha_movement_effects(effect_rows, config=None):
    """Summarize movement effects across participants by condition."""

    config = config or AlphaMovementAnalysisConfig()
    grouped = _group_rows(effect_rows, lambda row: (_clean_id(row.get("dataset")), row["trial_label"]))
    summary_rows = []
    for key, rows in sorted(grouped.items()):
        dataset, trial_label = key
        participants = {_clean_id(row.get("participant")) for row in rows}
        summary_row = {
            "dataset": dataset,
            "trial_label": trial_label,
            "n_participants": len(participants),
        }
        for metric in config.metrics:
            mean, std, sem = _summary_stats(_to_float(row.get(metric)) for row in rows)
            summary_row[f"{metric}_mean"] = mean
            summary_row[f"{metric}_std"] = std
            summary_row[f"{metric}_sem"] = sem
        summary_rows.append(summary_row)
    return summary_rows


def _selected_labels(rows, plot_labels):
    def label_key(value):
        numeric = _to_float(value)
        if np.isfinite(numeric):
            return (0, numeric, value)
        return (1, 0.0, value)

    labels = sorted(
        {_clean_id(row.get("trial_label")) for row in rows},
        key=label_key,
    )
    if plot_labels is None:
        return labels
    requested = {_clean_id(label) for label in plot_labels}
    return [label for label in labels if label in requested]


def _mean_timecourse(rows, metric, plot_labels):
    selected = set(_selected_labels(rows, plot_labels))
    grouped = defaultdict(list)
    for row in rows:
        label = _clean_id(row.get("trial_label"))
        if label in selected:
            grouped[(label, _to_float(row.get("time_s")))].append(_to_float(row.get(metric)))

    timecourses = defaultdict(list)
    for key, values in grouped.items():
        label, time_s = key
        mean, _, sem = _summary_stats(values)
        timecourses[label].append((time_s, mean, sem))
    return {label: sorted(values, key=lambda item: item[0]) for label, values in timecourses.items()}


def _plot_metric_timecourse(rows, metric, ylabel, output_path, plot_labels):
    timecourses = _mean_timecourse(rows, metric, plot_labels)
    figure, axes = plt.subplots(figsize=(8, 5))
    for label, values in timecourses.items():
        array = np.asarray(values, dtype=float)
        axes.plot(array[:, 0], array[:, 1], label=f"condition {label}")
        axes.fill_between(
            array[:, 0],
            array[:, 1] - array[:, 2],
            array[:, 1] + array[:, 2],
            alpha=0.15,
        )

    axes.axvline(0, color="black", linewidth=1, linestyle="--")
    axes.set_xlabel("time from stimulus (s)")
    axes.set_ylabel(ylabel)
    axes.legend(fontsize="small", ncol=2)
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_projected_trajectories(rows, output_path, plot_labels):
    labels = _selected_labels(rows, plot_labels)
    figure, axes = plt.subplots(figsize=(6, 6))
    for label in labels:
        label_rows = [row for row in rows if _clean_id(row.get("trial_label")) == label]
        grouped_by_time = _group_rows(label_rows, lambda row: _to_float(row.get("time_s")))
        points = []
        for time_s, time_rows in sorted(grouped_by_time.items()):
            points.append(
                (
                    time_s,
                    _mean(time_rows, "projected_x_mm"),
                    _mean(time_rows, "projected_y_mm"),
                )
            )
        array = np.asarray(points, dtype=float)
        axes.plot(array[:, 1], array[:, 2], marker=".", label=f"condition {label}")
        zero_index = int(np.argmin(np.abs(array[:, 0])))
        axes.scatter(array[zero_index, 1], array[zero_index, 2], s=25)

    axes.set_xlabel("projected x (mm)")
    axes.set_ylabel("projected y (mm)")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend(fontsize="small", ncol=2)
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_alpha_movement_plots(rows, output_dir, *, plot_labels=None):
    """Write condition-level movement plots from summary rows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_projected_trajectories(rows, output_dir / "alpha_movement_projected_trajectories.png", plot_labels)
    _plot_metric_timecourse(
        rows,
        "projected_speed_mm_per_s",
        "projected speed (mm/s)",
        output_dir / "alpha_movement_projected_speed.png",
        plot_labels,
    )
    _plot_metric_timecourse(
        rows,
        "displacement_mm",
        "3D displacement from first sample (mm)",
        output_dir / "alpha_movement_displacement.png",
        plot_labels,
    )


def export_alpha_movement_analysis(
    movement_summary_path,
    effect_output_path,
    condition_summary_output_path,
    *,
    plots_dir=None,
    config=None,
):
    """Load movement summaries and export pre/post effects and plots."""

    config = config or AlphaMovementAnalysisConfig()
    movement_rows = load_alpha_movement_rows(movement_summary_path)
    effect_rows = analyze_alpha_movement_windows(movement_rows, config)
    summary_rows = summarize_alpha_movement_effects(effect_rows, config)
    write_csv_rows(effect_rows, effect_output_path)
    write_csv_rows(summary_rows, condition_summary_output_path)
    if plots_dir:
        write_alpha_movement_plots(movement_rows, plots_dir, plot_labels=config.plot_labels)
    return effect_rows, summary_rows
