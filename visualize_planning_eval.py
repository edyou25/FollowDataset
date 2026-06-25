#!/usr/bin/env python3
"""Summarize and visualize planning.py -e JSONL records."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np


MODE_SPECS = (
    ("origin_planning", "Origin"),
    ("robot_safe_planning", "Robot Safe"),
    ("human_robot_safe_planning", "Human+Robot Safe"),
)


def _as_path(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _as_circles(value: Any) -> np.ndarray:
    rows = []
    for obstacle in value or []:
        if isinstance(obstacle, dict):
            rows.append(
                [
                    float(obstacle.get("x", 0.0)),
                    float(obstacle.get("y", 0.0)),
                    float(obstacle.get("r", 0.0)),
                ]
            )
        else:
            rows.append([float(obstacle[0]), float(obstacle[1]), float(obstacle[2])])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _as_segments(value: Any) -> np.ndarray:
    rows = []
    for segment in value or []:
        if isinstance(segment, dict):
            if "p1" in segment and "p2" in segment:
                p1 = segment["p1"]
                p2 = segment["p2"]
                rows.append([float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])])
            else:
                rows.append(
                    [
                        float(segment.get("x1", 0.0)),
                        float(segment.get("y1", 0.0)),
                        float(segment.get("x2", 0.0)),
                        float(segment.get("y2", 0.0)),
                    ]
                )
        else:
            rows.append(
                [float(segment[0]), float(segment[1]), float(segment[2]), float(segment[3])]
            )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def _point_segment_distances(points: np.ndarray, segment: np.ndarray) -> np.ndarray:
    p1 = segment[:2]
    p2 = segment[2:4]
    direction = p2 - p1
    denom = float(np.dot(direction, direction))
    if denom < 1e-12:
        return np.linalg.norm(points - p1[None, :], axis=1)
    t = np.sum((points - p1[None, :]) * direction[None, :], axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = p1[None, :] + t[:, None] * direction[None, :]
    return np.linalg.norm(points - closest, axis=1)


def collision_mask(
    path: Optional[np.ndarray],
    radius: float,
    circles: np.ndarray,
    segments: np.ndarray,
    margin: float,
) -> Optional[np.ndarray]:
    if path is None:
        return None
    mask = np.zeros(len(path), dtype=bool)
    effective_radius = float(radius) + float(margin)
    for obstacle in circles:
        distance = np.linalg.norm(path - obstacle[:2][None, :], axis=1)
        mask |= distance <= float(obstacle[2]) + effective_radius
    for segment in segments:
        mask |= _point_segment_distances(path, segment) <= effective_radius
    return mask


def rms_jerk(path: Optional[np.ndarray]) -> Optional[float]:
    """Match the repository scorer: RMS norm of the unscaled third difference."""
    if path is None or len(path) < 4:
        return None
    jerk = np.diff(np.diff(np.diff(path, axis=0), axis=0), axis=0)
    return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line_number, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if record.get("event") == "planning_eval":
                    record["_source_file"] = str(path)
                    records.append(record)
    if not records:
        raise ValueError("No planning_eval records found in the selected JSONL files.")
    return records


def _resolve_inputs(raw_inputs: list[str], logs_dir: Path) -> list[Path]:
    if not raw_inputs:
        candidates = sorted(
            (path for path in logs_dir.glob("planning_eval_*.jsonl") if path.stat().st_size > 0),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No non-empty planning_eval_*.jsonl files in {logs_dir}")
        return [candidates[-1]]

    paths = []
    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if path.exists():
            paths.append(path.resolve())
            continue
        matches = [Path(match) for match in sorted(glob.glob(raw))]
        if not matches:
            raise FileNotFoundError(f"Eval log or glob not found: {raw}")
        paths.extend(match.resolve() for match in matches if match.is_file())
    return paths


def compute_metrics(
    records: list[dict[str, Any]],
    collision_margin: float,
) -> list[dict[str, Any]]:
    rows = []
    for record_index, record in enumerate(records):
        environment_available = (
            "obstacles" in record and "segment_obstacles" in record
        )
        circles = _as_circles(record.get("obstacles"))
        segments = _as_segments(record.get("segment_obstacles"))
        robot_radius = float(record.get("robot_radius", 0.3))
        human_radius = float(record.get("human_radius", 0.3))

        for mode_key, mode_label in MODE_SPECS:
            planning = record.get(mode_key) or {}
            robot_path = _as_path(planning.get("robot_path", planning.get("path")))
            human_path = _as_path(planning.get("human_path"))
            robot_mask = (
                collision_mask(
                    robot_path, robot_radius, circles, segments, collision_margin
                )
                if environment_available
                else None
            )
            human_mask = (
                collision_mask(
                    human_path, human_radius, circles, segments, collision_margin
                )
                if environment_available
                else None
            )
            rows.append(
                {
                    "source_file": record.get("_source_file"),
                    "record_index": int(record_index),
                    "planning_index": int(record.get("planning_index", record_index)),
                    "frame": int(record.get("frame", 0)),
                    "mode": mode_key,
                    "mode_label": mode_label,
                    "robot_path_collision": (
                        bool(np.any(robot_mask)) if robot_mask is not None else None
                    ),
                    "human_path_collision": (
                        bool(np.any(human_mask)) if human_mask is not None else None
                    ),
                    "robot_collision_points": (
                        int(np.count_nonzero(robot_mask)) if robot_mask is not None else None
                    ),
                    "robot_path_points": (
                        int(len(robot_mask)) if robot_mask is not None else None
                    ),
                    "human_collision_points": (
                        int(np.count_nonzero(human_mask)) if human_mask is not None else None
                    ),
                    "human_path_points": (
                        int(len(human_mask)) if human_mask is not None else None
                    ),
                    "robot_path_point_collision_rate": (
                        float(np.mean(robot_mask)) if robot_mask is not None and len(robot_mask) else None
                    ),
                    "human_path_point_collision_rate": (
                        float(np.mean(human_mask)) if human_mask is not None and len(human_mask) else None
                    ),
                    "smoothness_rms_jerk": rms_jerk(robot_path),
                    "diffusion_time_ms": float(planning.get("diffusion_time_ms", 0.0)),
                    "qp_time_ms": float(planning.get("qp_time_ms", 0.0)),
                    "total_time_ms": float(planning.get("total_time_ms", 0.0)),
                }
            )
    return rows


def _mean_bool(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _point_rate(
    rows: list[dict[str, Any]],
    collision_key: str,
    total_key: str,
) -> Optional[float]:
    available = [
        row for row in rows if row.get(collision_key) is not None and row.get(total_key) is not None
    ]
    total = sum(int(row[total_key]) for row in available)
    if total <= 0:
        return None
    return float(sum(int(row[collision_key]) for row in available) / total)


def _distribution_summary(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode_key, mode_label in MODE_SPECS:
        mode_rows = [row for row in rows if row["mode"] == mode_key]
        smoothness = [
            float(row["smoothness_rms_jerk"])
            for row in mode_rows
            if row.get("smoothness_rms_jerk") is not None
        ]
        diffusion = [float(row["diffusion_time_ms"]) for row in mode_rows]
        qp = [float(row["qp_time_ms"]) for row in mode_rows]
        total = [float(row["total_time_ms"]) for row in mode_rows]
        summary[mode_key] = {
            "label": mode_label,
            "planning_count": len(mode_rows),
            "robot_collision_rate_path": _mean_bool(mode_rows, "robot_path_collision"),
            "human_collision_rate_path": _mean_bool(mode_rows, "human_path_collision"),
            "robot_collision_rate_path_point": _point_rate(
                mode_rows, "robot_collision_points", "robot_path_points"
            ),
            "human_collision_rate_path_point": _point_rate(
                mode_rows, "human_collision_points", "human_path_points"
            ),
            "smoothness_rms_jerk": _distribution_summary(smoothness),
            "diffusion_time_ms": _distribution_summary(diffusion),
            "qp_time_ms": _distribution_summary(qp),
            "total_time_ms": _distribution_summary(total),
        }
    return summary


def _write_csv(rows: list[dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rate_values(summary: dict[str, Any], key: str) -> list[float]:
    return [
        np.nan if summary[mode_key][key] is None else 100.0 * float(summary[mode_key][key])
        for mode_key, _label in MODE_SPECS
    ]


def _annotate_missing(ax, values: list[float]):
    if all(np.isnan(value) for value in values):
        ax.text(
            0.5,
            0.5,
            "Unavailable in this eval log.\nRun planning.py -e again with the updated logger.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#666666",
        )


def plot_metrics(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
    dpi: int,
    show: bool,
):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for _key, label in MODE_SPECS]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    robot_path_rates = _rate_values(summary, "robot_collision_rate_path")
    human_path_rates = _rate_values(summary, "human_collision_rate_path")
    axes[0, 0].bar(x - width / 2, robot_path_rates, width, label="Robot")
    axes[0, 0].bar(x + width / 2, human_path_rates, width, label="Human")
    axes[0, 0].set_title("Collision Rate per Planned Path")
    axes[0, 0].set_ylabel("Paths with collision (%)")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend()
    _annotate_missing(axes[0, 0], robot_path_rates + human_path_rates)

    robot_point_rates = _rate_values(summary, "robot_collision_rate_path_point")
    human_point_rates = _rate_values(summary, "human_collision_rate_path_point")
    axes[0, 1].bar(x - width / 2, robot_point_rates, width, label="Robot")
    axes[0, 1].bar(x + width / 2, human_point_rates, width, label="Human")
    axes[0, 1].set_title("Collision Rate per Planned Path Point")
    axes[0, 1].set_ylabel("Colliding path points (%)")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend()
    _annotate_missing(axes[0, 1], robot_point_rates + human_point_rates)

    smoothness_data = [
        [
            float(row["smoothness_rms_jerk"])
            for row in rows
            if row["mode"] == mode_key and row.get("smoothness_rms_jerk") is not None
        ]
        for mode_key, _label in MODE_SPECS
    ]
    if all(smoothness_data):
        axes[1, 0].boxplot(smoothness_data, labels=labels, showmeans=True)
    else:
        available = [(data, label) for data, label in zip(smoothness_data, labels) if data]
        if available:
            axes[1, 0].boxplot(
                [item[0] for item in available],
                labels=[item[1] for item in available],
                showmeans=True,
            )
        else:
            axes[1, 0].text(0.5, 0.5, "No valid paths", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Path Smoothness")
    axes[1, 0].set_ylabel("RMS jerk (lower is smoother)")

    origin_rows = [row for row in rows if row["mode"] == "origin_planning"]
    robot_rows = [row for row in rows if row["mode"] == "robot_safe_planning"]
    human_robot_rows = [row for row in rows if row["mode"] == "human_robot_safe_planning"]
    timing_data = [
        [float(row["diffusion_time_ms"]) for row in origin_rows],
        [float(row["qp_time_ms"]) for row in robot_rows],
        [float(row["qp_time_ms"]) for row in human_robot_rows],
    ]
    timing_labels = ["Diffusion", "Robot QP", "Human+Robot QP"]
    axes[1, 1].boxplot(timing_data, labels=timing_labels, showmeans=True)
    axes[1, 1].set_title("Planning Stage Time")
    axes[1, 1].set_ylabel("Time (ms)")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Planning Eval Metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _print_summary(summary: dict[str, Any]):
    print("-" * 108)
    print(
        f"{'Mode':<19} {'Robot path':>12} {'Human path':>12} "
        f"{'Robot points':>14} {'Human points':>14} {'RMS jerk':>12}"
    )
    print("-" * 108)
    for mode_key, mode_label in MODE_SPECS:
        item = summary[mode_key]

        def rate(key: str) -> str:
            value = item[key]
            return "N/A" if value is None else f"{100.0 * value:.2f}%"

        smoothness = item["smoothness_rms_jerk"]
        smoothness_text = "N/A" if smoothness is None else f"{smoothness['mean']:.6f}"
        print(
            f"{mode_label:<19} {rate('robot_collision_rate_path'):>12} "
            f"{rate('human_collision_rate_path'):>12} "
            f"{rate('robot_collision_rate_path_point'):>14} "
            f"{rate('human_collision_rate_path_point'):>14} "
            f"{smoothness_text:>12}"
        )


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize planning.py -e JSONL metrics")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Eval JSONL paths or glob patterns. Defaults to the latest non-empty eval log.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=script_dir / "logs",
        help="Directory searched when no input is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input_stem>_viz beside the first input.",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.1,
        help="Extra collision margin added to robot/human radii in meters.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window.")
    args = parser.parse_args()

    input_paths = _resolve_inputs(args.inputs, args.logs_dir)
    output_dir = args.output_dir or (
        input_paths[0].parent / f"{input_paths[0].stem}_viz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_paths)
    rows = compute_metrics(records, collision_margin=args.collision_margin)
    summary = summarize_metrics(rows)

    csv_path = output_dir / "planning_eval_metrics.csv"
    summary_path = output_dir / "planning_eval_summary.json"
    figure_path = output_dir / "planning_eval_overview.png"
    _write_csv(rows, csv_path)
    summary_path.write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in input_paths],
                "collision_margin": float(args.collision_margin),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_metrics(rows, summary, figure_path, dpi=args.dpi, show=args.show)

    _print_summary(summary)
    print(f"Records: {len(records)}")
    print(f"Figure : {figure_path}")
    print(f"Summary: {summary_path}")
    print(f"Metrics: {csv_path}")


if __name__ == "__main__":
    main()
