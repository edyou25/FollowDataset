#!/usr/bin/env python3
from __future__ import annotations

"""Detailed offline analysis for benchmark_results.json.

This script contains no ROS, detector, diffusion-policy, or safety-filter code.
It only reads the JSON produced by benchmark_generate.py and generates:

- detailed detection/interaction/planning time-series plots;
- overall and per-state statistics;
- path/clearance/intervention/runtime distributions;
- one detection + planning visualization for every planning frame;
- CSV tables and a lightweight HTML frame gallery.

The input JSON therefore acts as the immutable algorithm-run record, while all
analysis can be rerun without repeating DR-SPAAM or diffusion inference.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_ORDER = ("diffusion", "diffusion_qp", "safe_compliance", "ours")
MODE_LABELS = {
    "diffusion": "diffusion",
    "diffusion_qp": "diffusion+qp",
    "safe_compliance": "safe-compliance",
    "ours": "ours",
}
MODE_COLORS = {
    "diffusion": "tab:gray",
    "diffusion_qp": "tab:blue",
    "safe_compliance": "tab:orange",
    "ours": "tab:green",
}


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    version = int(data.get("schema_version", 0))
    if version < 2:
        raise ValueError(
            f"Unsupported benchmark schema_version={version}. "
            "Use the raw-result benchmark script that writes schema_version >= 2."
        )
    return data


def as_xy(value: Any) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=float)
    if array.size == 0:
        return np.zeros((0, 2), dtype=float)
    return array.reshape(-1, 2)


def as_xyz(value: Any) -> np.ndarray:
    array = np.asarray(value if value is not None else [], dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    return array.reshape(-1, 3)


def finite_array(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def numeric_stats(values: Iterable[Any]) -> dict[str, Optional[float]]:
    array = finite_array(values)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def bool_rate(values: Iterable[Any]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean([bool(v) for v in values]))


def path_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def smooth(values: pd.Series, window: int) -> pd.Series:
    window = max(1, int(window))
    if window <= 1:
        return values
    return values.rolling(window, center=True, min_periods=1).mean()


def build_detection_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in data.get("detection_frames", []):
        robot = np.asarray(frame.get("robot_position", [np.nan, np.nan]), dtype=float).reshape(-1)
        human_world = np.asarray(
            frame.get("human_world_position", [np.nan, np.nan]),
            dtype=float,
        ).reshape(-1)
        human_local = np.asarray(
            frame.get("human_local_position", [np.nan, np.nan]),
            dtype=float,
        ).reshape(-1)
        timing = frame.get("timing", {}) or {}
        row = {
            "scan_idx": frame.get("scan_idx"),
            "stamp": frame.get("stamp"),
            "relative_time": frame.get("relative_time"),
            "robot_x": robot[0] if len(robot) > 0 else np.nan,
            "robot_y": robot[1] if len(robot) > 1 else np.nan,
            "robot_yaw": frame.get("robot_yaw"),
            "human_world_x": human_world[0] if len(human_world) > 0 else np.nan,
            "human_world_y": human_world[1] if len(human_world) > 1 else np.nan,
            "human_local_x": human_local[0] if len(human_local) > 0 else np.nan,
            "human_local_y": human_local[1] if len(human_local) > 1 else np.nan,
            "accepted": bool(frame.get("accepted", False)),
            "fallback": bool(frame.get("fallback", False)),
            "fallback_kind": frame.get("fallback_kind", ""),
            "confidence": frame.get("confidence"),
            "raw_count": frame.get("raw_count"),
            "high_conf_count": frame.get("high_conf_count"),
            "rescue_count": frame.get("rescue_count"),
            "rear_count": frame.get("rear_count"),
            "tracker_reason": frame.get("tracker_reason", ""),
            "interaction_label": frame.get("interaction_label", "unknown"),
            "segmentation_raw_label": frame.get("segmentation_raw_label", "unknown"),
            "segmentation_samples": frame.get("segmentation_samples"),
            "segmentation_decode_ms": frame.get("segmentation_decode_ms"),
            "downstream_tracking_mode": frame.get("downstream_tracking_mode", ""),
            "downstream_kf_prediction": bool(frame.get("downstream_kf_prediction", False)),
            "downstream_kf_misses": frame.get("downstream_kf_misses"),
            "downstream_prior_fallback": bool(frame.get("downstream_prior_fallback", False)),
            "temporal_reset": bool(frame.get("temporal_reset", False)),
            "scan_preprocess_ms": timing.get("scan_preprocess_ms"),
            "detector_ms": timing.get("detector_ms"),
            "perception_pipeline_ms": timing.get("perception_pipeline_ms"),
            "frame_total_ms": timing.get("frame_total_ms"),
            "planning_plan_idx": frame.get("planning_plan_idx"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("relative_time").reset_index(drop=True)
        df["human_relative_distance"] = np.sqrt(
            df["human_local_x"].astype(float) ** 2
            + df["human_local_y"].astype(float) ** 2
        )
        dt = df["relative_time"].astype(float).diff()
        step = np.sqrt(
            df["human_local_x"].astype(float).diff() ** 2
            + df["human_local_y"].astype(float).diff() ** 2
        )
        df["human_relative_step"] = step
        df["human_relative_speed"] = step / dt.replace(0, np.nan)
        robot_step = np.sqrt(
            df["robot_x"].astype(float).diff() ** 2
            + df["robot_y"].astype(float).diff() ** 2
        )
        df["recorded_robot_speed"] = robot_step / dt.replace(0, np.nan)
    return df


def build_planning_df(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in data.get("planning_frames", []):
        timing = frame.get("timing", {}) or {}
        for mode in MODE_ORDER:
            payload = (frame.get("modes", {}) or {}).get(mode)
            if payload is None:
                continue
            rows.append(
                {
                    "plan_idx": frame.get("plan_idx"),
                    "scan_idx": frame.get("scan_idx"),
                    "stamp": frame.get("stamp"),
                    "relative_time": frame.get("relative_time"),
                    "interaction_label": frame.get("interaction_label", "unknown"),
                    "challenge_frame": bool(frame.get("challenge_frame", False)),
                    "detector_fallback": bool(frame.get("detector_fallback", False)),
                    "detector_confidence": frame.get("detector_confidence"),
                    "mode": mode,
                    "observation_ms": timing.get("observation_ms"),
                    "inference_ms": timing.get("inference_ms"),
                    "planning_preprocess_ms": timing.get("planning_preprocess_ms"),
                    "variant_evaluation_ms": timing.get("variant_evaluation_ms"),
                    "planning_pipeline_ms": timing.get("planning_pipeline_ms"),
                    "variant_runtime_ms": payload.get("runtime_ms"),
                    "total_mode_runtime_ms": (
                        (timing.get("inference_ms") or 0.0)
                        + (payload.get("runtime_ms") or 0.0)
                    ),
                    "projected_collision": bool(payload.get("projected_collision", False)),
                    "collision_who": payload.get("collision_who", ""),
                    "collision_step": payload.get("collision_step"),
                    "safety_violation": bool(payload.get("safety_violation", False)),
                    "robot_min_clearance": payload.get("robot_min_clearance"),
                    "human_min_clearance": payload.get("human_min_clearance"),
                    "combined_min_clearance": payload.get("combined_min_clearance"),
                    "mean_action_shift": payload.get("mean_action_shift"),
                    "mean_delta_shift": payload.get("mean_delta_shift"),
                    "strong_intervention": bool(payload.get("strong_intervention", False)),
                    "safety_modified_steps": payload.get("safety_modified_steps"),
                    "safety_constraint_count": payload.get("safety_constraint_count"),
                    "safety_min_clearance": payload.get("safety_min_clearance"),
                    "safety_intervention_fraction": payload.get("safety_intervention_fraction"),
                    "compliance_steps": payload.get("compliance_steps"),
                    "compliance_intervention_fraction": payload.get("compliance_intervention_fraction"),
                    "robot_horizon_displacement": payload.get("robot_horizon_displacement"),
                    "human_horizon_displacement": payload.get("human_horizon_displacement"),
                    "horizon_duration_sec": payload.get("horizon_duration_sec"),
                    "path_progress_m": payload.get("path_progress_m"),
                    "progress_speed_mps": payload.get("progress_speed_mps"),
                    "path_deviation": payload.get("path_deviation"),
                    "guide_unnecessary_intervention_m": payload.get("guide_unnecessary_intervention_m"),
                    "tether_response_error_m": payload.get("tether_response_error_m"),
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["relative_time", "mode"]).reset_index(drop=True)
    return df


def state_episodes(detection_df: pd.DataFrame) -> pd.DataFrame:
    if detection_df.empty:
        return pd.DataFrame(
            columns=["episode_idx", "label", "start_time", "end_time", "duration_sec", "frames"]
        )
    times = detection_df["relative_time"].astype(float).to_numpy()
    labels = detection_df["interaction_label"].astype(str).to_numpy()
    median_dt = float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.0
    rows = []
    start = 0
    ep = 0
    for idx in range(1, len(labels) + 1):
        if idx == len(labels) or labels[idx] != labels[start]:
            start_time = float(times[start])
            end_time = float(times[idx - 1] + median_dt)
            rows.append(
                {
                    "episode_idx": ep,
                    "label": labels[start],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_sec": max(0.0, end_time - start_time),
                    "frames": int(idx - start),
                }
            )
            ep += 1
            start = idx
    return pd.DataFrame(rows)


def mode_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"frames": 0}
    return {
        "frames": int(len(df)),
        "projected_collision_rate": bool_rate(df["projected_collision"]),
        "safety_violation_rate": bool_rate(df["safety_violation"]),
        "strong_intervention_rate": bool_rate(df["strong_intervention"]),
        "combined_min_clearance_m": numeric_stats(df["combined_min_clearance"]),
        "robot_min_clearance_m": numeric_stats(df["robot_min_clearance"]),
        "human_min_clearance_m": numeric_stats(df["human_min_clearance"]),
        "progress_speed_mps": numeric_stats(df["progress_speed_mps"]),
        "path_progress_m": numeric_stats(df["path_progress_m"]),
        "path_deviation_m": numeric_stats(df["path_deviation"]),
        "mean_delta_shift_m": numeric_stats(df["mean_delta_shift"]),
        "mean_action_shift": numeric_stats(df["mean_action_shift"]),
        "safety_intervention_fraction": numeric_stats(df["safety_intervention_fraction"]),
        "compliance_intervention_fraction": numeric_stats(df["compliance_intervention_fraction"]),
        "tether_response_error_m": numeric_stats(df["tether_response_error_m"]),
        "guide_unnecessary_intervention_m": numeric_stats(df["guide_unnecessary_intervention_m"]),
        "variant_runtime_ms": numeric_stats(df["variant_runtime_ms"]),
        "total_mode_runtime_ms": numeric_stats(df["total_mode_runtime_ms"]),
        "robot_horizon_displacement_m": numeric_stats(df["robot_horizon_displacement"]),
        "human_horizon_displacement_m": numeric_stats(df["human_horizon_displacement"]),
    }


def build_summary(
    data: dict[str, Any],
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
) -> dict[str, Any]:
    config = data.get("configuration", {}) or {}
    recorded_path = as_xy(data.get("recorded_robot_path", []))
    odom = np.asarray(data.get("recorded_odom_samples", []), dtype=float)
    if odom.size:
        odom = odom.reshape(-1, 4)
        odom_dt = np.diff(odom[:, 0])
        odom_ds = np.linalg.norm(np.diff(odom[:, 1:3], axis=0), axis=1)
        valid = odom_dt > 1e-6
        odom_speed = odom_ds[valid] / odom_dt[valid]
        raw_odom_length = float(np.sum(odom_ds))
    else:
        odom_speed = np.zeros((0,), dtype=float)
        raw_odom_length = 0.0

    relative_distance = (
        detection_df["human_relative_distance"]
        if "human_relative_distance" in detection_df
        else pd.Series(dtype=float)
    )

    detection_summary = {
        "frames": int(len(detection_df)),
        "accepted_rate": bool_rate(detection_df["accepted"]) if len(detection_df) else None,
        "fallback_rate": bool_rate(detection_df["fallback"]) if len(detection_df) else None,
        "downstream_kf_prediction_rate": (
            bool_rate(detection_df["downstream_kf_prediction"]) if len(detection_df) else None
        ),
        "confidence": numeric_stats(detection_df.get("confidence", [])),
        "raw_candidate_count": numeric_stats(detection_df.get("raw_count", [])),
        "high_conf_candidate_count": numeric_stats(detection_df.get("high_conf_count", [])),
        "rear_candidate_count": numeric_stats(detection_df.get("rear_count", [])),
        "human_local_x_m": numeric_stats(detection_df.get("human_local_x", [])),
        "human_local_y_m": numeric_stats(detection_df.get("human_local_y", [])),
        "human_relative_distance_m": numeric_stats(relative_distance),
        "human_relative_step_m": numeric_stats(detection_df.get("human_relative_step", [])),
        "human_relative_speed_mps": numeric_stats(detection_df.get("human_relative_speed", [])),
        "detector_ms": numeric_stats(detection_df.get("detector_ms", [])),
        "segmentation_decode_ms": numeric_stats(detection_df.get("segmentation_decode_ms", [])),
        "perception_pipeline_ms": numeric_stats(detection_df.get("perception_pipeline_ms", [])),
        "frame_total_ms": numeric_stats(detection_df.get("frame_total_ms", [])),
        "interaction_counts": (
            detection_df["interaction_label"].value_counts(dropna=False).to_dict()
            if len(detection_df)
            else {}
        ),
        "segmentation_raw_counts": (
            detection_df["segmentation_raw_label"].value_counts(dropna=False).to_dict()
            if len(detection_df)
            else {}
        ),
        "interaction_switches": max(0, int(len(episodes_df) - 1)),
        "interaction_episodes": episodes_df.to_dict(orient="records"),
    }

    planning_summary: dict[str, Any] = {
        "planning_frames": int(planning_df["plan_idx"].nunique()) if len(planning_df) else 0,
        "challenge_frames": int(
            planning_df.loc[
                planning_df["mode"] == "diffusion", "challenge_frame"
            ].sum()
        ) if len(planning_df) else 0,
        "timing": {},
        "modes": {},
    }
    if len(planning_df):
        base = planning_df[planning_df["mode"] == "diffusion"]
        planning_summary["timing"] = {
            "observation_ms": numeric_stats(base["observation_ms"]),
            "inference_ms": numeric_stats(base["inference_ms"]),
            "planning_preprocess_ms": numeric_stats(base["planning_preprocess_ms"]),
            "variant_evaluation_ms": numeric_stats(base["variant_evaluation_ms"]),
            "planning_pipeline_ms": numeric_stats(base["planning_pipeline_ms"]),
        }
        for mode in MODE_ORDER:
            mode_df = planning_df[planning_df["mode"] == mode]
            planning_summary["modes"][mode] = {
                "all": mode_summary(mode_df),
                "guide": mode_summary(
                    mode_df[mode_df["interaction_label"] == "guide"]
                ),
                "tether": mode_summary(
                    mode_df[mode_df["interaction_label"] == "tether"]
                ),
                "challenge": mode_summary(
                    mode_df[mode_df["challenge_frame"]]
                ),
            }

    summary = {
        "source_json": data.get("bag", {}).get("path"),
        "schema_version": data.get("schema_version"),
        "bag": data.get("bag", {}),
        "configuration": {
            "safety_eval_clearance": config.get("safety_eval_clearance"),
            "challenge_clearance_threshold": config.get("challenge_clearance_threshold"),
            "strong_intervention_threshold": config.get("strong_intervention_threshold"),
            "leash_length": config.get("leash_length"),
            "rear_sector_angle_deg": config.get("rear_sector_angle_deg"),
            "plan_every": config.get("plan_every"),
        },
        "recorded_path": {
            "reference_points": int(len(recorded_path)),
            "reference_length_m": path_length(recorded_path),
            "raw_odom_length_m": raw_odom_length,
            "raw_odom_speed_mps": numeric_stats(odom_speed),
            "scan_sampled_robot_speed_mps": numeric_stats(
                detection_df.get("recorded_robot_speed", [])
            ),
        },
        "detection": detection_summary,
        "planning": planning_summary,
    }

    if planning_summary.get("modes"):
        modes = planning_summary["modes"]
        def mean_value(mode: str, state: str, metric: str) -> Optional[float]:
            try:
                return modes[mode][state][metric]["mean"]
            except Exception:
                return None

        ours_guide = mean_value("ours", "guide", "progress_speed_mps")
        safe_guide = mean_value("safe_compliance", "guide", "progress_speed_mps")
        dq_tether = mean_value("diffusion_qp", "tether", "tether_response_error_m")
        ours_tether = mean_value("ours", "tether", "tether_response_error_m")
        comparison = {}
        if ours_guide is not None and safe_guide not in (None, 0):
            comparison["ours_vs_safe_compliance_guide_speed_ratio"] = float(
                ours_guide / safe_guide
            )
        if dq_tether is not None and ours_tether is not None:
            comparison["diffusion_qp_minus_ours_tether_response_error_m"] = float(
                dq_tether - ours_tether
            )
        summary["comparisons"] = comparison

    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state_spans(ax: plt.Axes, detection_df: pd.DataFrame) -> None:
    if detection_df.empty:
        return
    episodes = state_episodes(detection_df)
    for _, ep in episodes.iterrows():
        label = str(ep["label"])
        if label == "tether":
            ax.axvspan(
                float(ep["start_time"]),
                float(ep["end_time"]),
                alpha=0.07,
                color="tab:red",
                linewidth=0,
            )


def finalize_time_axes(axes: Iterable[plt.Axes], detection_df: pd.DataFrame) -> None:
    for ax in axes:
        state_spans(ax, detection_df)
        ax.grid(True, alpha=0.28)
        ax.set_xlabel("Time [s]")


def plot_detection_timeline(
    df: pd.DataFrame,
    output_path: Path,
    *,
    leash_length: float,
    smooth_window: int,
) -> None:
    if df.empty:
        return
    t = df["relative_time"].astype(float)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

    axes[0].plot(t, smooth(df["human_local_x"].astype(float), smooth_window), label="human local x")
    axes[0].plot(t, smooth(df["human_local_y"].astype(float), smooth_window), label="human local y")
    axes[0].axhline(-float(leash_length), linestyle="--", linewidth=1, label="-leash length")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title("Detected human relative position")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(t, df["confidence"].astype(float), label="target confidence")
    fallback = df["fallback"].astype(bool).to_numpy()
    accepted = df["accepted"].astype(bool).to_numpy()
    axes[1].scatter(t[accepted], df.loc[accepted, "confidence"], s=8, label="accepted")
    axes[1].scatter(t[fallback], df.loc[fallback, "confidence"], s=10, marker="x", label="fallback")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Detector/tracker acceptance")
    axes[1].legend(frameon=False, ncol=3)

    for key, label in (
        ("raw_count", "raw"),
        ("high_conf_count", "high-conf"),
        ("rescue_count", "rescue"),
        ("rear_count", "rear"),
    ):
        axes[2].plot(t, df[key].astype(float), label=label)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Detection candidate counts")
    axes[2].legend(frameon=False, ncol=4)

    axes[3].plot(t, df["detector_ms"].astype(float), label="DR-SPAAM")
    axes[3].plot(t, df["segmentation_decode_ms"].astype(float), label="segmentation decode")
    axes[3].plot(t, df["perception_pipeline_ms"].astype(float), label="perception pipeline")
    axes[3].set_ylabel("Time [ms]")
    axes[3].set_title("Perception runtime")
    axes[3].legend(frameon=False, ncol=3)

    finalize_time_axes(axes, df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_interaction_timeline(
    df: pd.DataFrame,
    output_path: Path,
    *,
    smooth_window: int,
) -> None:
    if df.empty:
        return
    t = df["relative_time"].astype(float)
    label_num = (df["interaction_label"].astype(str) == "tether").astype(int)
    raw_num = (df["segmentation_raw_label"].astype(str) == "tether").astype(int)

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    axes[0].step(t, label_num, where="post", label="final state")
    axes[0].step(t, raw_num, where="post", alpha=0.65, label="raw segmentation")
    axes[0].set_yticks([0, 1], labels=["guide", "tether"])
    axes[0].set_title("Guide/tether state timeline")
    axes[0].legend(frameon=False)

    axes[1].plot(
        t,
        smooth(df["human_relative_distance"].astype(float), smooth_window),
        label="robot-human relative distance",
    )
    axes[1].set_ylabel("Distance [m]")
    axes[1].set_title("Relative distance")

    axes[2].plot(
        t,
        smooth(df["human_relative_speed"].astype(float), smooth_window),
        label="relative speed",
    )
    axes[2].set_ylabel("Speed [m/s]")
    axes[2].set_title("Relative motion rate")

    axes[3].plot(t, df["segmentation_samples"].astype(float), label="window samples")
    axes[3].set_ylabel("Samples")
    axes[3].set_title("Segmentation window fill")

    finalize_time_axes(axes, df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def pivot_mode(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.pivot_table(
        index="relative_time",
        columns="mode",
        values=key,
        aggfunc="first",
    ).sort_index()


def plot_runtime_timeline(
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
    *,
    smooth_window: int,
) -> None:
    if planning_df.empty:
        return
    base = planning_df[planning_df["mode"] == "diffusion"].copy()
    t = base["relative_time"].astype(float)
    runtime = pivot_mode(planning_df, "variant_runtime_ms")
    total_runtime = pivot_mode(planning_df, "total_mode_runtime_ms")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, smooth(base["observation_ms"].astype(float), smooth_window), label="observation")
    axes[0].plot(t, smooth(base["inference_ms"].astype(float), smooth_window), label="diffusion inference")
    axes[0].plot(t, smooth(base["planning_preprocess_ms"].astype(float), smooth_window), label="preprocess")
    axes[0].set_ylabel("Time [ms]")
    axes[0].set_title("Shared planning stages")
    axes[0].legend(frameon=False, ncol=3)

    for mode in MODE_ORDER:
        if mode in runtime:
            axes[1].plot(
                runtime.index,
                smooth(runtime[mode].astype(float), smooth_window),
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
            )
    axes[1].set_ylabel("Time [ms]")
    axes[1].set_title("Variant evaluation runtime")
    axes[1].legend(frameon=False, ncol=4)

    for mode in MODE_ORDER:
        if mode in total_runtime:
            axes[2].plot(
                total_runtime.index,
                smooth(total_runtime[mode].astype(float), smooth_window),
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
            )
    axes[2].set_ylabel("Time [ms]")
    axes[2].set_title("Inference + variant runtime")
    axes[2].legend(frameon=False, ncol=4)

    axes[3].plot(
        t,
        smooth(base["planning_pipeline_ms"].astype(float), smooth_window),
        label="whole planning frame",
    )
    if not detection_df.empty:
        axes[3].plot(
            detection_df["relative_time"],
            smooth(detection_df["frame_total_ms"].astype(float), smooth_window),
            alpha=0.7,
            label="whole scan pipeline",
        )
    axes[3].set_ylabel("Time [ms]")
    axes[3].set_title("End-to-end offline processing time")
    axes[3].legend(frameon=False)

    finalize_time_axes(axes, detection_df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_clearance_timeline(
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
    *,
    safety_threshold: float,
    challenge_threshold: float,
    smooth_window: int,
) -> None:
    if planning_df.empty:
        return
    combined = pivot_mode(planning_df, "combined_min_clearance")
    robot = pivot_mode(planning_df, "robot_min_clearance")
    human = pivot_mode(planning_df, "human_min_clearance")
    base = planning_df[planning_df["mode"] == "diffusion"].copy()

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for ax, pivot, title in (
        (axes[0], combined, "Combined minimum clearance"),
        (axes[1], robot, "Robot minimum clearance"),
        (axes[2], human, "Human minimum clearance"),
    ):
        for mode in MODE_ORDER:
            if mode in pivot:
                ax.plot(
                    pivot.index,
                    smooth(pivot[mode].astype(float), smooth_window),
                    label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode],
                )
        ax.axhline(safety_threshold, linestyle="--", linewidth=1, label="safety envelope")
        ax.set_ylabel("Clearance [m]")
        ax.set_title(title)
    axes[0].axhline(challenge_threshold, linestyle=":", linewidth=1, label="challenge threshold")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].legend(frameon=False, ncol=5)

    challenge = base["challenge_frame"].astype(int)
    axes[3].step(base["relative_time"], challenge, where="post", label="challenge frame")
    for mode in MODE_ORDER:
        subset = planning_df[planning_df["mode"] == mode]
        axes[3].step(
            subset["relative_time"],
            subset["safety_violation"].astype(int),
            where="post",
            alpha=0.65,
            label=f"{MODE_LABELS[mode]} violation",
            color=MODE_COLORS[mode],
        )
    axes[3].set_yticks([0, 1])
    axes[3].set_ylabel("Flag")
    axes[3].set_title("Challenge and safety-envelope flags")
    axes[3].legend(frameon=False, ncol=3)

    finalize_time_axes(axes, detection_df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_progress_timeline(
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
    *,
    smooth_window: int,
) -> None:
    if planning_df.empty:
        return
    metrics = (
        ("progress_speed_mps", "Progress speed [m/s]", "Predicted path progress speed"),
        ("path_deviation", "Deviation [m]", "Predicted path deviation from recorded robot reference"),
        ("robot_horizon_displacement", "Distance [m]", "Robot horizon displacement"),
        ("human_horizon_displacement", "Distance [m]", "Human horizon displacement"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for ax, (key, ylabel, title) in zip(axes, metrics):
        pivot = pivot_mode(planning_df, key)
        for mode in MODE_ORDER:
            if mode in pivot:
                ax.plot(
                    pivot.index,
                    smooth(pivot[mode].astype(float), smooth_window),
                    label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode],
                )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=4)
    finalize_time_axes(axes, detection_df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_intervention_timeline(
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
    *,
    smooth_window: int,
) -> None:
    if planning_df.empty:
        return
    metrics = (
        ("mean_delta_shift", "Shift [m]", "Action-sequence delta modification"),
        ("mean_action_shift", "Action shift", "Policy action modification"),
        ("safety_intervention_fraction", "Fraction", "Safety-filter intervention fraction"),
        ("compliance_intervention_fraction", "Fraction", "Compliance intervention fraction"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for ax, (key, ylabel, title) in zip(axes, metrics):
        pivot = pivot_mode(planning_df, key)
        for mode in MODE_ORDER:
            if mode in pivot:
                ax.plot(
                    pivot.index,
                    smooth(pivot[mode].astype(float), smooth_window),
                    label=MODE_LABELS[mode],
                    color=MODE_COLORS[mode],
                )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=4)
    finalize_time_axes(axes, detection_df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_compliance_timeline(
    detection_df: pd.DataFrame,
    planning_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if planning_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)

    tether = planning_df[planning_df["interaction_label"] == "tether"]
    for mode in MODE_ORDER:
        subset = tether[tether["mode"] == mode]
        if len(subset):
            axes[0].plot(
                subset["relative_time"],
                subset["tether_response_error_m"].astype(float),
                marker=".",
                markersize=3,
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
            )
    axes[0].set_ylabel("Error [m]")
    axes[0].set_title("Tether response deviation from full-compliance response")
    axes[0].legend(frameon=False, ncol=4)

    guide = planning_df[planning_df["interaction_label"] == "guide"]
    for mode in MODE_ORDER:
        subset = guide[guide["mode"] == mode]
        if len(subset):
            axes[1].plot(
                subset["relative_time"],
                subset["guide_unnecessary_intervention_m"].astype(float),
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
            )
    axes[1].set_ylabel("Shift [m]")
    axes[1].set_title("Guide-mode action modification")
    axes[1].legend(frameon=False, ncol=4)

    finalize_time_axes(axes, detection_df)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectory_overview(
    data: dict[str, Any],
    detection_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if detection_df.empty:
        return
    reference = as_xy(data.get("recorded_robot_path", []))
    robot = detection_df[["robot_x", "robot_y"]].to_numpy(float)
    human = detection_df[["human_world_x", "human_world_y"]].to_numpy(float)
    labels = detection_df["interaction_label"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
    if len(reference):
        ax.plot(reference[:, 0], reference[:, 1], "--", linewidth=1.2, color="0.5", label="recorded robot reference")
    ax.plot(robot[:, 0], robot[:, 1], linewidth=1.5, color="black", label="recorded robot")

    used: set[str] = set()
    start = 0
    while start < len(human):
        end = start + 1
        while end < len(human) and labels[end] == labels[start]:
            end += 1
        label = labels[start]
        color = "tab:red" if label == "tether" else "tab:green"
        legend_label = f"detected human: {label}"
        ax.plot(
            human[start:end, 0],
            human[start:end, 1],
            linewidth=2,
            color=color,
            label=legend_label if label not in used else None,
        )
        used.add(label)
        start = end

    ax.scatter(robot[0, 0], robot[0, 1], s=45, marker="o", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Recorded robot and detected human trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def grouped_mean(
    planning_df: pd.DataFrame,
    mode: str,
    key: str,
    *,
    state: Optional[str] = None,
    challenge_only: bool = False,
) -> float:
    df = planning_df[planning_df["mode"] == mode]
    if state is not None:
        df = df[df["interaction_label"] == state]
    if challenge_only:
        df = df[df["challenge_frame"]]
    values = pd.to_numeric(df[key], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def plot_overall_comparison(
    data: dict[str, Any],
    planning_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if planning_df.empty:
        return
    metrics = [
        (
            "challenge_safety",
            "Safety-envelope violation rate\n(challenging frames) ↓",
            lambda m: bool_rate(
                planning_df[
                    (planning_df["mode"] == m)
                    & planning_df["challenge_frame"]
                ]["safety_violation"]
            ),
        ),
        (
            "tether_error",
            "Tether response error [m] ↓",
            lambda m: grouped_mean(planning_df, m, "tether_response_error_m", state="tether"),
        ),
        (
            "guide_intervention",
            "Guide unnecessary intervention [m] ↓",
            lambda m: grouped_mean(
                planning_df,
                m,
                "guide_unnecessary_intervention_m",
                state="guide",
            ),
        ),
        (
            "guide_speed",
            "Guide progress speed [m/s] ↑",
            lambda m: grouped_mean(planning_df, m, "progress_speed_mps", state="guide"),
        ),
        (
            "clearance",
            "Mean combined minimum clearance [m] ↑",
            lambda m: grouped_mean(planning_df, m, "combined_min_clearance"),
        ),
        (
            "runtime",
            "Mean inference + variant runtime [ms] ↓",
            lambda m: grouped_mean(planning_df, m, "total_mode_runtime_ms"),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (_, title, fn) in zip(axes.flat, metrics):
        vals = [fn(mode) for mode in MODE_ORDER]
        ax.bar(
            [MODE_LABELS[m] for m in MODE_ORDER],
            vals,
            color=[MODE_COLORS[m] for m in MODE_ORDER],
            alpha=0.86,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", alpha=0.28)
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            ax.text(
                0.5,
                0.5,
                "No applicable frames",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            for i, value in enumerate(vals):
                if np.isfinite(value):
                    ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    challenge_count = int(
        planning_df.loc[
            planning_df["mode"] == "diffusion", "challenge_frame"
        ].sum()
    )
    guide_count = int(
        (
            (planning_df["mode"] == "diffusion")
            & (planning_df["interaction_label"] == "guide")
        ).sum()
    )
    tether_count = int(
        (
            (planning_df["mode"] == "diffusion")
            & (planning_df["interaction_label"] == "tether")
        ).sum()
    )
    fig.suptitle(
        f"Static real-world benchmark | challenge={challenge_count}, "
        f"guide={guide_count}, tether={tether_count}"
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_distributions(
    planning_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if planning_df.empty:
        return
    metrics = [
        ("total_mode_runtime_ms", "Inference + variant runtime [ms]"),
        ("combined_min_clearance", "Combined minimum clearance [m]"),
        ("progress_speed_mps", "Progress speed [m/s]"),
        ("path_deviation", "Path deviation [m]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        arrays = []
        labels = []
        for mode in MODE_ORDER:
            values = pd.to_numeric(
                planning_df.loc[planning_df["mode"] == mode, key],
                errors="coerce",
            ).dropna().to_numpy()
            arrays.append(values)
            labels.append(MODE_LABELS[mode])
        ax.boxplot(arrays, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", alpha=0.28)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def draw_rear_sector(
    ax: plt.Axes,
    *,
    aperture_deg: float,
    min_range: float,
    max_range: float,
) -> None:
    center = math.pi
    half = math.radians(float(aperture_deg)) / 2.0
    angles = np.linspace(center - half, center + half, 80)
    outer = np.column_stack([max_range * np.cos(angles), max_range * np.sin(angles)])
    inner = np.column_stack([min_range * np.cos(angles[::-1]), min_range * np.sin(angles[::-1])])
    polygon = np.vstack([outer, inner])
    ax.fill(
        polygon[:, 0],
        polygon[:, 1],
        alpha=0.06,
        color="tab:blue",
        linewidth=0,
        label="rear search sector",
    )
    for angle in (center - half, center + half):
        ax.plot(
            [0.0, max_range * math.cos(angle)],
            [0.0, max_range * math.sin(angle)],
            "--",
            linewidth=0.8,
            color="0.5",
        )


def frame_metrics_text(frame: dict[str, Any]) -> str:
    lines = []
    for mode in MODE_ORDER:
        p = (frame.get("modes", {}) or {}).get(mode, {})
        if not p:
            continue
        clearance = p.get("combined_min_clearance")
        speed = p.get("progress_speed_mps")
        shift = p.get("mean_delta_shift")
        runtime = p.get("runtime_ms")
        def f(v: Any, digits: int = 3) -> str:
            try:
                x = float(v)
                return f"{x:.{digits}f}" if np.isfinite(x) else "nan"
            except Exception:
                return "nan"
        lines.append(
            f"{MODE_LABELS[mode]:16s} "
            f"clr={f(clearance)}m  "
            f"prog={f(speed)}m/s  "
            f"shift={f(shift)}m  "
            f"rt={f(runtime, 2)}ms"
        )
    return "\n".join(lines)


def plot_frame(
    data: dict[str, Any],
    frame: dict[str, Any],
    detection_frame: Optional[dict[str, Any]],
    output_path: Path,
    *,
    frame_dpi: int,
    world_radius: float,
) -> None:
    config = data.get("configuration", {}) or {}
    reference = as_xy(data.get("recorded_robot_path", []))
    robot = np.asarray(frame.get("robot_position", [0, 0]), dtype=float).reshape(2)
    human = np.asarray(frame.get("human_position", [0, 0]), dtype=float).reshape(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)

    ax = axes[0]
    if detection_frame is not None:
        scan = as_xy(detection_frame.get("scan_points_local", []))
        if len(scan):
            ax.scatter(scan[:, 0], scan[:, 1], s=2, alpha=0.32, label="LaserScan")
        det = detection_frame.get("detector", {}) or {}
        raw = as_xy(det.get("raw_detections_xy", []))
        rear = as_xy(det.get("rear_detections_xy", []))
        target = det.get("target_xy")
        if len(raw):
            ax.scatter(raw[:, 0], raw[:, 1], s=22, marker="x", label="raw detections")
        if len(rear):
            ax.scatter(rear[:, 0], rear[:, 1], s=34, facecolors="none", edgecolors="tab:orange", label="rear candidates")
        if target is not None:
            target_arr = np.asarray(target, dtype=float).reshape(2)
            ax.scatter(target_arr[0], target_arr[1], s=80, marker="*", label="tracker target")
        downstream = np.asarray(
            detection_frame.get("human_local_position", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        if np.isfinite(downstream).all():
            ax.scatter(
                downstream[0],
                downstream[1],
                s=55,
                marker="s",
                label="downstream human state",
            )
    draw_rear_sector(
        ax,
        aperture_deg=float(config.get("rear_sector_angle_deg", 90.0)),
        min_range=float(config.get("rear_min_range", 0.3)),
        max_range=float(config.get("rear_max_range", 3.0)),
    )
    ax.scatter(0, 0, s=50, marker="s", color="black", label="robot")
    ax.set_aspect("equal", adjustable="box")
    max_range = float(config.get("rear_max_range", 3.0))
    ax.set_xlim(-max_range - 0.4, max_range + 0.4)
    ax.set_ylim(-max_range - 0.4, max_range + 0.4)
    ax.set_xlabel("Detector x-forward [m]")
    ax.set_ylabel("Detector y-right [m]")
    ax.set_title("Detection / tracking")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    obstacles = as_xyz(frame.get("obstacles_world", []))
    if len(obstacles):
        distance = np.linalg.norm(obstacles[:, :2] - robot[None, :], axis=1)
        display = obstacles[distance <= world_radius * 1.5]
        if len(display):
            ax.scatter(
                display[:, 0],
                display[:, 1],
                s=4,
                alpha=0.3,
                color="0.55",
                label="safety obstacle points",
            )
    if len(reference):
        dist = np.linalg.norm(reference - robot[None, :], axis=1)
        ref_local = reference[dist <= world_radius * 1.8]
        if len(ref_local):
            ax.plot(
                ref_local[:, 0],
                ref_local[:, 1],
                "--",
                linewidth=1.0,
                color="0.5",
                label="recorded robot reference",
            )

    ax.scatter(robot[0], robot[1], s=60, marker="s", color="black", label="current robot", zorder=10)
    ax.scatter(human[0], human[1], s=60, marker="o", color="tab:red", label="current human", zorder=10)

    for mode in MODE_ORDER:
        payload = (frame.get("modes", {}) or {}).get(mode, {})
        robot_path = as_xy(payload.get("robot_path", []))
        human_path = as_xy(payload.get("human_path", []))
        if len(robot_path):
            ax.plot(
                robot_path[:, 0],
                robot_path[:, 1],
                linewidth=2.0,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
        if len(human_path):
            ax.plot(
                human_path[:, 0],
                human_path[:, 1],
                linewidth=1.0,
                linestyle=":",
                color=MODE_COLORS[mode],
                alpha=0.7,
            )

    ax.set_xlim(robot[0] - world_radius, robot[0] + world_radius)
    ax.set_ylim(robot[1] - world_radius, robot[1] + world_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_title("Counterfactual planning rollouts")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")

    fallback = bool(frame.get("detector_fallback", False))
    fig.suptitle(
        f"plan={frame.get('plan_idx')} | scan={frame.get('scan_idx')} | "
        f"t={float(frame.get('relative_time', 0.0)):.2f}s | "
        f"state={frame.get('interaction_label')} | "
        f"challenge={bool(frame.get('challenge_frame', False))} | "
        f"det={'fallback' if fallback else 'accepted'}"
    )
    fig.text(
        0.5,
        0.01,
        frame_metrics_text(frame),
        ha="center",
        va="bottom",
        family="monospace",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(frame_dpi), bbox_inches="tight")
    plt.close(fig)


def write_gallery(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cards = []
    for row in rows:
        image_name = html.escape(row["image"])
        cards.append(
            "<figure>"
            f'<a href="{image_name}"><img src="{image_name}" loading="lazy"></a>'
            f"<figcaption>plan {row['plan_idx']} · t={row['relative_time']:.2f}s · "
            f"{html.escape(str(row['interaction_label']))} · "
            f"challenge={row['challenge_frame']}</figcaption>"
            "</figure>"
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>Benchmark frame gallery</title>
<style>
body{font-family:sans-serif;margin:20px;background:#fafafa;color:#222}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
figure{margin:0;background:white;border:1px solid #ddd;padding:8px;border-radius:8px}
img{width:100%;height:auto;display:block}
figcaption{font-size:13px;margin-top:6px}
</style>
<h1>Benchmark frame gallery</h1>
<div class="grid">
""" + "\n".join(cards) + "\n</div>\n"
    output_path.write_text(document, encoding="utf-8")


def generate_frame_visualizations(
    data: dict[str, Any],
    output_dir: Path,
    *,
    frame_step: int,
    frame_start: int,
    frame_end: int,
    frame_dpi: int,
    world_radius: float,
) -> pd.DataFrame:
    detection_by_scan = {
        int(frame.get("scan_idx", -1)): frame
        for frame in data.get("detection_frames", [])
    }
    planning = data.get("planning_frames", [])
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame in planning:
        plan_idx = int(frame.get("plan_idx", -1))
        if plan_idx < frame_start:
            continue
        if frame_end >= 0 and plan_idx > frame_end:
            continue
        if ((plan_idx - frame_start) % max(1, frame_step)) != 0:
            continue
        scan_idx = int(frame.get("scan_idx", -1))
        filename = f"frame_{plan_idx:05d}_t{float(frame.get('relative_time', 0.0)):07.2f}.png"
        path = frames_dir / filename
        plot_frame(
            data,
            frame,
            detection_by_scan.get(scan_idx),
            path,
            frame_dpi=frame_dpi,
            world_radius=world_radius,
        )
        rows.append(
            {
                "plan_idx": plan_idx,
                "scan_idx": scan_idx,
                "relative_time": float(frame.get("relative_time", 0.0)),
                "interaction_label": frame.get("interaction_label", "unknown"),
                "challenge_frame": bool(frame.get("challenge_frame", False)),
                "image": f"frames/{filename}",
            }
        )
        if len(rows) % 50 == 0:
            print(f"      rendered {len(rows)} frame visualizations")
    gallery_path = output_dir / "frame_gallery.html"
    write_gallery(rows, gallery_path)
    return pd.DataFrame(rows)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_json.expanduser().resolve()
    data = load_json(input_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading raw benchmark: {input_path}")
    detection_df = build_detection_df(data)
    planning_df = build_planning_df(data)
    episodes_df = state_episodes(detection_df)
    print(
        f"      detection={len(detection_df)}, "
        f"planning={planning_df['plan_idx'].nunique() if len(planning_df) else 0}"
    )

    print("[2/5] Computing detailed statistics")
    summary = build_summary(data, detection_df, planning_df, episodes_df)

    detection_csv = output_dir / "detection_metrics.csv"
    planning_csv = output_dir / "planning_metrics.csv"
    episodes_csv = output_dir / "interaction_episodes.csv"
    detection_df.to_csv(detection_csv, index=False)
    planning_df.to_csv(planning_csv, index=False)
    episodes_df.to_csv(episodes_csv, index=False)

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2, allow_nan=False)

    config = data.get("configuration", {}) or {}
    safety_threshold = float(
        args.safety_eval_clearance
        if args.safety_eval_clearance is not None
        else config.get("safety_eval_clearance", config.get("safety_margin", 0.2))
    )
    challenge_threshold = float(
        args.challenge_clearance_threshold
        if args.challenge_clearance_threshold is not None
        else config.get("challenge_clearance_threshold", safety_threshold)
    )
    leash_length = float(config.get("leash_length", 1.2))

    print("[3/5] Plotting whole-sequence time curves")
    plot_detection_timeline(
        detection_df,
        output_dir / "01_detection_timeline.png",
        leash_length=leash_length,
        smooth_window=args.smooth_window,
    )
    plot_interaction_timeline(
        detection_df,
        output_dir / "02_interaction_timeline.png",
        smooth_window=args.smooth_window,
    )
    plot_runtime_timeline(
        detection_df,
        planning_df,
        output_dir / "03_runtime_timeline.png",
        smooth_window=args.smooth_window,
    )
    plot_clearance_timeline(
        detection_df,
        planning_df,
        output_dir / "04_clearance_timeline.png",
        safety_threshold=safety_threshold,
        challenge_threshold=challenge_threshold,
        smooth_window=args.smooth_window,
    )
    plot_progress_timeline(
        detection_df,
        planning_df,
        output_dir / "05_progress_path_timeline.png",
        smooth_window=args.smooth_window,
    )
    plot_intervention_timeline(
        detection_df,
        planning_df,
        output_dir / "06_intervention_timeline.png",
        smooth_window=args.smooth_window,
    )
    plot_compliance_timeline(
        detection_df,
        planning_df,
        output_dir / "07_compliance_timeline.png",
    )
    plot_trajectory_overview(
        data,
        detection_df,
        output_dir / "08_trajectory_overview.png",
    )
    plot_overall_comparison(
        data,
        planning_df,
        output_dir / "09_overall_comparison.png",
    )
    plot_distributions(
        planning_df,
        output_dir / "10_metric_distributions.png",
    )

    print("[4/5] Rendering per-frame detection/planning visualizations")
    if args.no_frame_images:
        frame_df = pd.DataFrame()
        print("      skipped by --no-frame-images")
    else:
        frame_df = generate_frame_visualizations(
            data,
            output_dir,
            frame_step=args.frame_step,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_dpi=args.frame_dpi,
            world_radius=args.world_radius,
        )
        frame_df.to_csv(output_dir / "frame_index.csv", index=False)

    print("[5/5] Analysis complete")
    artifacts = {
        "analysis_summary": str(output_dir / "analysis_summary.json"),
        "detection_metrics_csv": str(detection_csv),
        "planning_metrics_csv": str(planning_csv),
        "interaction_episodes_csv": str(episodes_csv),
        "detection_timeline": str(output_dir / "01_detection_timeline.png"),
        "interaction_timeline": str(output_dir / "02_interaction_timeline.png"),
        "runtime_timeline": str(output_dir / "03_runtime_timeline.png"),
        "clearance_timeline": str(output_dir / "04_clearance_timeline.png"),
        "progress_path_timeline": str(output_dir / "05_progress_path_timeline.png"),
        "intervention_timeline": str(output_dir / "06_intervention_timeline.png"),
        "compliance_timeline": str(output_dir / "07_compliance_timeline.png"),
        "trajectory_overview": str(output_dir / "08_trajectory_overview.png"),
        "overall_comparison": str(output_dir / "09_overall_comparison.png"),
        "metric_distributions": str(output_dir / "10_metric_distributions.png"),
        "frame_gallery": (
            str(output_dir / "frame_gallery.html")
            if not args.no_frame_images
            else None
        ),
        "frame_count_rendered": int(len(frame_df)),
    }
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return {"summary": summary, "artifacts": artifacts}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detailed analysis/visualization of benchmark_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "tests/artifacts/2026-08-02-18-18-16.bag/"
            "benchmark_results.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <input-json-parent>/analysis",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered rolling mean window for time curves; 1 keeps raw values",
    )
    parser.add_argument(
        "--safety-eval-clearance",
        type=float,
        default=None,
        help="Only changes analysis reference lines; raw benchmark flags remain unchanged",
    )
    parser.add_argument(
        "--challenge-clearance-threshold",
        type=float,
        default=None,
        help="Only changes plotted reference line; raw challenge flags remain unchanged",
    )
    parser.add_argument(
        "--no-frame-images",
        action="store_true",
        help="Skip per-planning-frame PNG generation",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument(
        "--frame-end",
        type=int,
        default=-1,
        help="-1 means through the last planning frame",
    )
    parser.add_argument("--frame-dpi", type=int, default=150)
    parser.add_argument(
        "--world-radius",
        type=float,
        default=3.0,
        help="Half-width of world-frame per-frame planning visualization [m]",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if args.frame_dpi < 50:
        raise ValueError("--frame-dpi must be >= 50")
    if args.world_radius <= 0:
        raise ValueError("--world-radius must be > 0")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_analysis(args)


if __name__ == "__main__":
    main()