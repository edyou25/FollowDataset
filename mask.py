#!/usr/bin/env python3
"""为专家数据生成规则事件掩码（速度/加速度/朝向突变）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# 支持从 data/ 目录直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_storage import DataStorage


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="same")


def temporal_dilate(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return mask
    pad = width // 2
    padded = np.pad(mask.astype(np.uint8), (pad, pad), mode="edge")
    window = np.ones(width, dtype=np.uint8)
    dilated = np.convolve(padded, window, mode="valid") > 0
    return dilated[: mask.shape[0]]


def compute_kinematics(path: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回速度向量、速度标量、加速度向量（逐帧对齐）。"""
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[dt <= 1e-4] = 1e-4  # 防止除零

    dpos = np.diff(path, axis=0, prepend=path[[0]])
    vel = dpos / dt[:, None]
    speed = np.linalg.norm(vel, axis=1)

    dvel = np.diff(vel, axis=0, prepend=vel[[0]])
    acc = dvel / dt[:, None]
    return vel, speed, acc


def detect_events(
    path: np.ndarray,
    timestamps: np.ndarray,
    cfg: Dict[str, float],
) -> Dict[str, np.ndarray]:
    vel, speed, acc = compute_kinematics(path, timestamps)

    speed_smooth = moving_average(speed, int(cfg["smooth_window"]))
    speed_jump = np.abs(speed - speed_smooth) > cfg["speed_jump_thresh"]

    acc_norm = np.linalg.norm(acc, axis=1)
    acc_event = acc_norm > cfg["acc_thresh"]

    heading = np.arctan2(vel[:, 1], vel[:, 0])
    heading_unwrapped = np.unwrap(heading)
    dheading = np.abs(np.diff(heading_unwrapped, prepend=heading_unwrapped[0]))
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[dt <= 1e-4] = 1e-4
    heading_rate = dheading / dt
    heading_event = (heading_rate > cfg["heading_rate_thresh"]) & (speed > cfg["min_speed_for_heading"])

    raw_mask = speed_jump | acc_event | heading_event
    mask = temporal_dilate(raw_mask, int(cfg["dilate_width"]))
    mask = merge_close_events(mask, path, cfg["merge_gap_dist"])

    onsets = mask & np.concatenate(([False], ~mask[:-1]))
    return {
        "mask": mask,
        "raw": raw_mask,
        "onsets": onsets,
        "speed": speed,
        "acc_norm": acc_norm,
        "heading_rate": heading_rate,
        "cause_speed_jump": speed_jump,
        "cause_acc_event": acc_event,
        "cause_heading_event": heading_event,
    }


def save_mask(episode_dir: Path, mask_dict: Dict[str, np.ndarray], cfg: Dict[str, float], out_name: str) -> Path:
    out_path = episode_dir / out_name
    np.savez(
        out_path,
        mask=mask_dict["mask"],
        raw=mask_dict["raw"],
        onsets=mask_dict["onsets"],
        speed=mask_dict["speed"],
        acc_norm=mask_dict["acc_norm"],
        heading_rate=mask_dict["heading_rate"],
        cause_speed_jump=mask_dict["cause_speed_jump"],
        cause_acc_event=mask_dict["cause_acc_event"],
        cause_heading_event=mask_dict["cause_heading_event"],
        cfg=cfg,
    )
    return out_path


def summarize(mask: np.ndarray, timestamps: np.ndarray, episode: str) -> str:
    onsets = mask & np.concatenate(([False], ~mask[:-1]))
    active_ratio = mask.mean() * 100
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    return (
        f"{episode}: events={onsets.sum()}  active={active_ratio:.1f}%  "
        f"frames={len(mask)}  duration={duration:.1f}s"
    )


def merge_close_events(mask: np.ndarray, path: np.ndarray, dist_thresh: float) -> np.ndarray:
    """Merge two events if spatial gap between end and next start is below dist_thresh (meters)."""
    if not mask.any():
        return mask
    merged = mask.copy()
    segments = []
    in_seg = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_seg:
            start = i
            in_seg = True
        elif not val and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask) - 1))

    for i in range(len(segments) - 1):
        end_curr = segments[i][1]
        start_next = segments[i + 1][0]
        if start_next <= end_curr + 1:
            continue
        if end_curr >= len(path) or start_next >= len(path):
            continue
        gap_dist = np.linalg.norm(path[end_curr] - path[start_next])
        if gap_dist < dist_thresh:
            merged[end_curr + 1 : start_next] = True

    # Remove very short segments after merge (length < dist_thresh)
    cleaned = merged.copy()
    in_seg = False
    start = 0
    for i, val in enumerate(merged):
        if val and not in_seg:
            start = i
            in_seg = True
        elif not val and in_seg:
            length = np.linalg.norm(path[start] - path[i - 1])
            if length < dist_thresh:
                cleaned[start:i] = False
            in_seg = False
    if in_seg:
        length = np.linalg.norm(path[start] - path[len(merged) - 1])
        if length < dist_thresh:
            cleaned[start:] = False
    return cleaned
    return merged


def main():
    parser = argparse.ArgumentParser(description="生成规则事件掩码（速度/加速度/朝向突变检测）")
    parser.add_argument("-d", "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data", help="数据目录，包含 episode_*")
    parser.add_argument("-e", "--episode", help="指定单个 episode 名称；不填则处理全部")
    parser.add_argument("--out-name", default="event_mask_rule_v0.npz", help="输出文件名（保存到每个 episode 目录下）")
    parser.add_argument("--speed-jump-thresh", type=float, default=0.35, help="速度相对平滑值的跳变阈值 (m/s)")
    parser.add_argument("--acc-thresh", type=float, default=10.0, help="加速度模长阈值 (m/s^2)")
    parser.add_argument("--heading-rate-thresh", type=float, default=1.0, help="朝向变化率阈值 (rad/s)")
    parser.add_argument("--min-speed-for-heading", type=float, default=0.2, help="朝向判断的最低速度 (m/s)")
    parser.add_argument("--smooth-window", type=int, default=9, help="速度平滑窗口（帧数，奇数更佳）")
    parser.add_argument("--dilate-width", type=int, default=7, help="事件掩码时间膨胀窗口（帧数）")
    parser.add_argument("--merge-gap-dist", type=float, default=1.0, help="若两段事件间路径距离小于该值（米），合并事件")
    args = parser.parse_args()

    episodes = [args.episode] if args.episode else DataStorage.list_episodes(str(args.data_dir))
    if not episodes:
        raise SystemExit(f"No episodes found in {args.data_dir}")

    cfg = {
        "speed_jump_thresh": args.speed_jump_thresh,
        "acc_thresh": args.acc_thresh,
        "heading_rate_thresh": args.heading_rate_thresh,
        "min_speed_for_heading": args.min_speed_for_heading,
        "smooth_window": args.smooth_window,
        "dilate_width": args.dilate_width,
        "merge_gap_dist": args.merge_gap_dist,
    }

    for ep in episodes:
        ep_dir = args.data_dir / ep
        if not ep_dir.exists():
            print(f"[skip] {ep_dir} not found")
            continue
        data = DataStorage.load_episode(str(ep_dir))
        robot_path = data["robot_path"]
        timestamps = data["timestamps"]

        mask_dict = detect_events(robot_path, timestamps, cfg)
        out_path = save_mask(ep_dir, mask_dict, cfg, args.out_name)
        print(f"Saved: {out_path} | {summarize(mask_dict['mask'], timestamps, ep)}")


if __name__ == "__main__":
    main()
