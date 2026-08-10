#!/usr/bin/env python3
"""Analyze human_detection_*.jsonl produced by the guide robot runtime."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def nested(record: dict, path: str, default: Any = None) -> Any:
    value: Any = record
    for key in path.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        elif isinstance(value, (list, tuple)) and key.isdigit():
            index = int(key)
            if index < 0 or index >= len(value):
                return default
            value = value[index]
        else:
            return default
    return value


def read_jsonl(paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[warn] {path}:{line_no}: invalid JSON: {exc}")
                    continue
                record["_source_file"] = str(path)
                record["_source_line"] = int(line_no)
                records.append(record)
    return records


def finite_series(values: Iterable[Any]) -> pd.Series:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    return series[np.isfinite(series)]


def stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    series = finite_series(values)
    if series.empty:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(series.size),
        "mean": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def percentage(count: int, total: int) -> float:
    return 100.0 * float(count) / float(total) if total > 0 else 0.0


def flatten_event(record: dict) -> dict:
    selected = record.get("selected_candidate") or {}
    after = record.get("tracking_after") or {}
    status = record.get("detector_status") or {}
    detector_frame = record.get("detector_frame") or {}
    rosbag = record.get("rosbag") or {}
    robot = record.get("robot") or {}
    row = {
        "source_file": record.get("_source_file"),
        "source_line": record.get("_source_line"),
        "record_index": record.get("record_index"),
        "refresh_index": record.get("refresh_index"),
        "frame": record.get("frame"),
        "ros_time": record.get("ros_time"),
        "wall_time": record.get("wall_time"),
        "duration_ms": record.get("duration_ms"),
        "outcome": record.get("outcome"),
        "success": record.get("success"),
        "reason": record.get("reason"),
        "detector_seq": detector_frame.get("seq", status.get("seq")),
        "detector_message_stamp": status.get("message_stamp"),
        "detector_receive_stamp": status.get("receive_stamp"),
        "detector_age": status.get("age"),
        "detector_count_callback": status.get("count"),
        "transform_error": status.get("transform_error"),
        "raw_candidate_count": record.get("raw_candidate_count", 0),
        "roi_candidate_count": record.get("roi_candidate_count", 0),
        "roi_kind": nested(record, "roi.kind"),
        "roi_range": nested(record, "roi.range"),
        "roi_angle_deg": nested(record, "roi.total_angle_deg"),
        "selected_raw_index": selected.get("raw_index"),
        "selected_x": nested(selected, "world_xy.0"),
        "selected_y": nested(selected, "world_xy.1"),
        "selected_robot_distance": selected.get("robot_distance"),
        "selected_rear_angle_error_deg": selected.get("rear_angle_error_deg"),
        "selected_mahalanobis_sq": selected.get("mahalanobis_sq"),
        "selected_jump": selected.get("jump_from_prediction"),
        "selected_sim_distance": selected.get("sim_distance"),
        "selected_score": selected.get("score"),
        "accepted_by_gate": selected.get("accepted_by_gate"),
        "accepted_by_jump_override": selected.get("accepted_by_jump_override"),
        "tracking_mode": after.get("mode"),
        "consecutive_misses": after.get("consecutive_misses"),
        "using_prediction": after.get("using_prediction"),
        "detector_rejected": after.get("detector_rejected"),
        "waiting": after.get("waiting"),
        "tracked_x": nested(after, "tracked_position.0"),
        "tracked_y": nested(after, "tracked_position.1"),
        "tracked_vx": nested(after, "tracked_velocity.0"),
        "tracked_vy": nested(after, "tracked_velocity.1"),
        "robot_x": nested(robot, "position.0"),
        "robot_y": nested(robot, "position.1"),
        "robot_heading_deg": robot.get("heading_deg"),
        "rosbag_reset_pending": rosbag.get("reset_pending"),
        "rosbag_reset_count": rosbag.get("reset_count"),
        "odom_replay_epoch": nested(record, "odom_replay.replay_epoch"),
    }
    return row


def flatten_candidates(record: dict) -> list[dict]:
    rows = []
    for candidate in record.get("candidate_scores") or []:
        row = {
            "source_file": record.get("_source_file"),
            "record_index": record.get("record_index"),
            "refresh_index": record.get("refresh_index"),
            "frame": record.get("frame"),
            "outcome": record.get("outcome"),
            "detector_seq": nested(record, "detector_frame.seq"),
            "candidate_index": candidate.get("candidate_index"),
            "raw_index": candidate.get("raw_index"),
            "x": nested(candidate, "world_xy.0"),
            "y": nested(candidate, "world_xy.1"),
            "robot_distance": candidate.get("robot_distance"),
            "rear_angle_error_deg": candidate.get("rear_angle_error_deg"),
            "mahalanobis_sq": candidate.get("mahalanobis_sq"),
            "gate_pass": candidate.get("gate_pass"),
            "jump_from_prediction": candidate.get("jump_from_prediction"),
            "jump_pass": candidate.get("jump_pass"),
            "sim_distance": candidate.get("sim_distance"),
            "score": candidate.get("score"),
            "plausible": candidate.get("plausible"),
            "selected": candidate.get("selected", False),
        }
        rows.append(row)
    return rows


def build_diagnosis(events: pd.DataFrame, outcomes: Counter) -> list[str]:
    diagnosis: list[str] = []
    total = len(events)
    new_frames = events[~events["outcome"].isin([
        "same_detector_frame_prediction",
        "same_detector_frame_stopped",
    ])]
    new_total = len(new_frames)

    raw_positive = new_frames[new_frames["raw_candidate_count"] > 0]
    roi_rejected = raw_positive[raw_positive["roi_candidate_count"] == 0]
    roi_ratio = len(roi_rejected) / max(1, len(raw_positive))
    if roi_ratio >= 0.30:
        diagnosis.append(
            f"ROI rejection is high: {len(roi_rejected)}/{len(raw_positive)} "
            f"({roi_ratio * 100:.1f}%) frames had candidates, but none remained. "
            "Check detector axes/TF first, then sector angle and range."
        )

    outlier_count = sum(
        count for name, count in outcomes.items() if "detector_outlier" in name
    )
    accepted_count = outcomes.get("detector_fused", 0) + outcomes.get(
        "track_initialized", 0
    )
    scored_total = accepted_count + outlier_count
    if scored_total > 0 and outlier_count / scored_total >= 0.20:
        diagnosis.append(
            f"KF outlier ratio is high: {outlier_count}/{scored_total} "
            f"({outlier_count / scored_total * 100:.1f}%). Check timestamp/TF "
            "alignment before loosening human_kf_gate or human_track_max_jump."
        )

    no_fresh = sum(
        count for name, count in outcomes.items() if name.startswith("no_fresh_detection")
    )
    if total > 0 and no_fresh / total >= 0.10:
        diagnosis.append(
            f"No-fresh-detection events occupy {no_fresh}/{total} "
            f"({no_fresh / total * 100:.1f}%). Compare detector rate with "
            "human_detection_timeout and inspect TF errors."
        )

    same_count = outcomes.get("same_detector_frame_prediction", 0)
    if total > 0 and same_count / total >= 0.50:
        diagnosis.append(
            f"{same_count}/{total} refreshes reused the same detector frame. "
            "This is normal when the control loop is faster than the detector, "
            "provided stale/no-fresh events remain low."
        )

    stop_count = sum(count for name, count in outcomes.items() if "stopped" in name)
    if stop_count > 0:
        diagnosis.append(
            f"Tracking stopped {stop_count} times. Inspect the corresponding "
            "reason, transform_error, raw/ROI counts, and simulation validity."
        )

    rosbag_resets = int(events["rosbag_reset_count"].max()) if not events.empty else 0
    if rosbag_resets > 0:
        diagnosis.append(
            f"Detected {rosbag_resets} rosbag loop reset(s). Reacquisition "
            "records should be treated as replay boundaries, not detector faults."
        )

    if new_total == 0:
        diagnosis.append("No new detector frames were logged.")
    if not diagnosis:
        diagnosis.append(
            "No dominant failure mode was detected from the available records."
        )
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze detailed guide-robot human detection JSONL logs."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON analysis outputs.",
    )
    args = parser.parse_args()

    missing = [path for path in args.logs if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    records = read_jsonl(args.logs)
    diagnostics = [
        record
        for record in records
        if record.get("event") == "human_detection_diagnostic"
    ]
    systems = [
        record
        for record in records
        if record.get("event") == "human_detection_system"
    ]
    if not diagnostics:
        raise RuntimeError("No human_detection_diagnostic records found")

    event_rows = [flatten_event(record) for record in diagnostics]
    candidate_rows = [
        row for record in diagnostics for row in flatten_candidates(record)
    ]
    events = pd.DataFrame(event_rows)
    candidates = pd.DataFrame(candidate_rows)
    outcomes = Counter(events["outcome"].fillna("unknown").astype(str))

    tracked_xy = events[["tracked_x", "tracked_y"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_track = tracked_xy.notna().all(axis=1)
    track_steps = np.linalg.norm(
        np.diff(tracked_xy[valid_track].to_numpy(dtype=float), axis=0),
        axis=1,
    ) if int(valid_track.sum()) >= 2 else np.array([], dtype=float)

    summary = {
        "input_files": [str(path) for path in args.logs],
        "diagnostic_records": int(len(events)),
        "system_records": int(len(systems)),
        "unique_detector_sequences": int(
            pd.to_numeric(events["detector_seq"], errors="coerce").nunique()
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "outcome_percent": {
            name: percentage(count, len(events))
            for name, count in sorted(outcomes.items())
        },
        "raw_candidate_count": stats(events["raw_candidate_count"]),
        "roi_candidate_count": stats(events["roi_candidate_count"]),
        "detector_age_sec": stats(events["detector_age"]),
        "processing_duration_ms": stats(events["duration_ms"]),
        "selected_mahalanobis_sq": stats(events["selected_mahalanobis_sq"]),
        "selected_jump_m": stats(events["selected_jump"]),
        "selected_sim_distance_m": stats(events["selected_sim_distance"]),
        "tracked_step_distance_m": stats(track_steps),
        "candidate_rows": int(len(candidates)),
        "rosbag_loop_resets": int(
            pd.to_numeric(events["rosbag_reset_count"], errors="coerce")
            .fillna(0)
            .max()
        ),
    }
    summary["diagnosis"] = build_diagnosis(events, outcomes)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.logs[0].parent / (
            args.logs[0].stem + "_analysis"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "human_detection_events.csv"
    candidates_path = output_dir / "human_detection_candidates.csv"
    outcomes_path = output_dir / "human_detection_outcomes.csv"
    summary_path = output_dir / "human_detection_summary.json"

    events.to_csv(events_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "outcome": name,
                "count": count,
                "percent": percentage(count, len(events)),
            }
            for name, count in outcomes.most_common()
        ]
    ).to_csv(outcomes_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Human detection analysis")
    print("=" * 72)
    print(f"Diagnostic records : {len(events)}")
    print(f"Unique detector seq: {summary['unique_detector_sequences']}")
    print(f"Candidate rows      : {len(candidates)}")
    print(f"Rosbag loop resets  : {summary['rosbag_loop_resets']}")
    print("\nOutcomes:")
    for name, count in outcomes.most_common():
        print(f"  {name:38s} {count:7d}  {percentage(count, len(events)):6.2f}%")
    print("\nKey statistics:")
    for name in [
        "raw_candidate_count",
        "roi_candidate_count",
        "detector_age_sec",
        "selected_mahalanobis_sq",
        "selected_jump_m",
        "tracked_step_distance_m",
        "processing_duration_ms",
    ]:
        item = summary[name]
        print(
            f"  {name:30s} count={item['count']:5d} "
            f"mean={item['mean']} p50={item['p50']} "
            f"p95={item['p95']} max={item['max']}"
        )
    print("\nDiagnosis:")
    for item in summary["diagnosis"]:
        print(f"  - {item}")
    print("\nOutputs:")
    print(f"  {events_path}")
    print(f"  {candidates_path}")
    print(f"  {outcomes_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()