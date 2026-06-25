#!/usr/bin/env python3
"""Convert FollowDataset episodes to the guidedog mocap CSV layout.

Target CSV layout copied from:
01-Sighted/11.csv

Columns:
    "", "Unnamed: 0", "robot_x", "robot_y", "robot_orientation",
    "robot_occlusion", "human_x", "human_y", "human_rotation",
    "human_occlusion", "state"

The FollowDataset positions are in meters. The guidedog mocap files use
millimeter-scale coordinates, so the default position scale is 1000.0.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_INPUT_DIR = Path("/home/yyf/IROS2026/FollowDataset/logs")
DEFAULT_OUTPUT_DIR = Path("/home/yyf/IROS2026/converted_guidedog_csv")

CSV_HEADER = [
    "",
    "Unnamed: 0",
    "robot_x",
    "robot_y",
    "robot_orientation",
    "robot_occlusion",
    "human_x",
    "human_y",
    "human_rotation",
    "human_occlusion",
    "state",
]


class ZarrReadError(RuntimeError):
    pass


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _decode_chunk(raw: bytes, compressor: dict | None) -> bytes:
    if compressor is None:
        return raw

    compressor_id = compressor.get("id")
    if compressor_id != "blosc":
        raise ZarrReadError(f"unsupported zarr compressor: {compressor_id!r}")

    try:
        import blosc
    except ImportError as exc:
        raise ZarrReadError(
            "this zarr store uses blosc compression; install python-blosc "
            "or run in an environment where `import blosc` works"
        ) from exc

    return blosc.decompress(raw)


def _fill_value_for(meta: dict, dtype: np.dtype) -> np.ndarray:
    fill_value = meta.get("fill_value", 0)
    if fill_value is None:
        fill_value = 0
    return np.asarray(fill_value, dtype=dtype)


def read_zarr_array(array_dir: Path) -> np.ndarray:
    """Read a simple zarr v2 array without requiring the zarr package."""

    meta_path = array_dir / ".zarray"
    if not meta_path.exists():
        raise ZarrReadError(f"missing zarr metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("zarr_format", 2)) != 2:
        raise ZarrReadError(f"only zarr v2 is supported: {array_dir}")
    if meta.get("filters"):
        raise ZarrReadError(f"zarr filters are not supported: {array_dir}")

    shape = tuple(int(v) for v in meta["shape"])
    chunks = tuple(int(v) for v in meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    order = meta.get("order", "C")
    separator = meta.get("dimension_separator", ".")
    compressor = meta.get("compressor")

    if len(shape) != len(chunks):
        raise ZarrReadError(f"invalid zarr shape/chunks in {array_dir}")

    out = np.empty(shape, dtype=dtype, order=order)
    out[...] = _fill_value_for(meta, dtype)

    chunk_grid = [
        range(_ceildiv(dim, chunk_dim)) for dim, chunk_dim in zip(shape, chunks)
    ]
    for chunk_index in itertools.product(*chunk_grid):
        chunk_name = separator.join(str(i) for i in chunk_index)
        chunk_path = array_dir / chunk_name
        if not chunk_path.exists():
            continue

        decoded = _decode_chunk(chunk_path.read_bytes(), compressor)
        flat = np.frombuffer(decoded, dtype=dtype)

        valid_shape = tuple(
            min(chunk_dim, dim - chunk_i * chunk_dim)
            for chunk_i, chunk_dim, dim in zip(chunk_index, chunks, shape)
        )
        full_size = int(np.prod(chunks, dtype=np.int64))
        valid_size = int(np.prod(valid_shape, dtype=np.int64))

        if flat.size == full_size:
            chunk = flat.reshape(chunks, order=order)
            source_slices = tuple(slice(0, n) for n in valid_shape)
        elif flat.size == valid_size:
            chunk = flat.reshape(valid_shape, order=order)
            source_slices = tuple(slice(None) for _ in valid_shape)
        else:
            raise ZarrReadError(
                f"unexpected chunk size in {chunk_path}: got {flat.size}, "
                f"expected {full_size} or {valid_size}"
            )

        dest_slices = tuple(
            slice(chunk_i * chunk_dim, chunk_i * chunk_dim + valid_dim)
            for chunk_i, chunk_dim, valid_dim in zip(
                chunk_index, chunks, valid_shape
            )
        )
        out[dest_slices] = chunk[source_slices]

    return out


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"expected quaternion array with shape (N, 4), got {quat.shape}")

    norms = np.linalg.norm(quat, axis=1)
    safe = norms > 1e-12
    normalized = np.zeros_like(quat, dtype=np.float64)
    normalized[safe] = quat[safe] / norms[safe, None]

    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]
    w = normalized[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw[~safe] = 0.0
    return yaw


def yaw_from_path(xy: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"expected xy path with shape (N, 2), got {xy.shape}")

    n = xy.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    delta = np.empty_like(xy)
    delta[:-1] = xy[1:] - xy[:-1]
    delta[-1] = xy[-1] - xy[-2]

    valid = np.linalg.norm(delta, axis=1) > eps
    if not np.any(valid):
        return np.zeros(n, dtype=np.float64)

    yaw = np.zeros(n, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    valid_yaw = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
    yaw[:] = np.interp(np.arange(n), valid_indices, valid_yaw)
    return _wrap_to_pi(yaw)


def choose_yaw(xy: np.ndarray, pose: np.ndarray | None) -> np.ndarray:
    """Use pose quaternion yaw when it carries signal; otherwise use path yaw."""

    path_yaw = yaw_from_path(xy)
    if pose is None or pose.ndim != 2 or pose.shape[1] < 7:
        return path_yaw

    quat_yaw = yaw_from_quat_xyzw(pose[:, 3:7])
    unwrapped = np.unwrap(quat_yaw)
    has_quat_signal = (
        np.nanmax(np.abs(unwrapped)) > 1e-8
        or (np.nanmax(unwrapped) - np.nanmin(unwrapped)) > 1e-8
    )
    return quat_yaw if has_quat_signal else path_yaw


def read_episode_arrays(
    trajectory_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"trajectory.zarr not found: {trajectory_dir}")

    robot_pose = None
    human_pose = None

    robot_pose_dir = trajectory_dir / "robot_base_pose"
    human_pose_dir = trajectory_dir / "human_base_pose"
    if robot_pose_dir.exists():
        robot_pose = np.asarray(read_zarr_array(robot_pose_dir), dtype=np.float64)
    if human_pose_dir.exists():
        human_pose = np.asarray(read_zarr_array(human_pose_dir), dtype=np.float64)

    if robot_pose is not None and robot_pose.ndim == 2 and robot_pose.shape[1] >= 2:
        robot_xy = robot_pose[:, :2]
    else:
        robot_xy = np.asarray(read_zarr_array(trajectory_dir / "robot_path"), dtype=np.float64)

    if human_pose is not None and human_pose.ndim == 2 and human_pose.shape[1] >= 2:
        human_xy = human_pose[:, :2]
    else:
        human_xy = np.asarray(read_zarr_array(trajectory_dir / "human_path"), dtype=np.float64)

    robot_yaw = choose_yaw(robot_xy, robot_pose)
    human_yaw = choose_yaw(human_xy, human_pose)
    state_dir = trajectory_dir / "state"
    if state_dir.exists():
        state = np.asarray(read_zarr_array(state_dir), dtype=np.int8).reshape(-1)
    else:
        state = np.full((len(robot_xy),), 2, dtype=np.int8)

    n = min(len(robot_xy), len(human_xy), len(robot_yaw), len(human_yaw), len(state))
    if n == 0:
        raise ValueError(f"empty episode: {trajectory_dir}")

    return robot_xy[:n], human_xy[:n], robot_yaw[:n], human_yaw[:n], state[:n]


def _format_float(value: float) -> str:
    return repr(float(value))


def iter_csv_rows(
    robot_xy: np.ndarray,
    human_xy: np.ndarray,
    robot_yaw: np.ndarray,
    human_yaw: np.ndarray,
    state: np.ndarray,
    *,
    position_scale: float,
    start_index: int = 0,
) -> Iterable[list[str]]:
    n = min(len(robot_xy), len(human_xy), len(robot_yaw), len(human_yaw), len(state))
    for local_i in range(n):
        row_i = start_index + local_i
        yield [
            str(row_i),
            str(row_i),
            _format_float(robot_xy[local_i, 0] * position_scale),
            _format_float(robot_xy[local_i, 1] * position_scale),
            _format_float(robot_yaw[local_i]),
            "False",
            _format_float(human_xy[local_i, 0] * position_scale),
            _format_float(human_xy[local_i, 1] * position_scale),
            _format_float(human_yaw[local_i]),
            "False",
            str(int(state[local_i])),
        ]


def write_episode_csv(
    episode_dir: Path,
    output_csv: Path,
    *,
    position_scale: float,
    start_index: int = 0,
) -> int:
    robot_xy, human_xy, robot_yaw, human_yaw, state = read_episode_arrays(
        episode_dir / "trajectory.zarr"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        count = 0
        for row in iter_csv_rows(
            robot_xy,
            human_xy,
            robot_yaw,
            human_yaw,
            state,
            position_scale=position_scale,
            start_index=start_index,
        ):
            writer.writerow(row)
            count += 1
    return count


def episode_dirs(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        p
        for p in input_dir.glob(pattern)
        if p.is_dir() and (p / "trajectory.zarr").exists()
    )


def write_combined_csv(
    episodes: Sequence[Path],
    output_csv: Path,
    *,
    position_scale: float,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for ep_dir in episodes:
            robot_xy, human_xy, robot_yaw, human_yaw, state = read_episode_arrays(
                ep_dir / "trajectory.zarr"
            )
            for row in iter_csv_rows(
                robot_xy,
                human_xy,
                robot_yaw,
                human_yaw,
                state,
                position_scale=position_scale,
                start_index=total,
            ):
                writer.writerow(row)
                total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FollowDataset/data episode zarr files to the guidedog "
            "mocap CSV schema used by 01-Sighted/11.csv."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"FollowDataset data directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "directory for per-episode CSV output "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=None,
        help="optional single CSV output path; when set, episodes are concatenated",
    )
    parser.add_argument(
        "--episode-pattern",
        default="episode_*",
        help="glob pattern under input-dir (default: episode_*)",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=1000.0,
        help="scale applied to x/y positions; 1000 converts meters to millimeters",
    )
    parser.add_argument(
        "--numeric-names",
        action="store_true",
        help="name per-episode CSV files as 0.csv, 1.csv, ... instead of episode_name.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert only the first N episodes, useful for a quick check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    episodes = episode_dirs(input_dir, args.episode_pattern)
    if args.limit is not None:
        episodes = episodes[: max(0, args.limit)]

    if not episodes:
        raise SystemExit(f"no episodes found in {input_dir} matching {args.episode_pattern!r}")

    if args.combined_csv is not None:
        output_csv = args.combined_csv.expanduser().resolve()
        total = write_combined_csv(
            episodes,
            output_csv,
            position_scale=float(args.position_scale),
        )
        print(f"wrote {total} rows from {len(episodes)} episodes to {output_csv}")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    total = 0
    for index, ep_dir in enumerate(episodes):
        name = f"{index}.csv" if args.numeric_names else f"{ep_dir.name}.csv"
        out_csv = output_dir / name
        rows = write_episode_csv(
            ep_dir,
            out_csv,
            position_scale=float(args.position_scale),
        )
        total += rows
        print(f"wrote {rows} rows: {out_csv}")

    print(f"done: {len(episodes)} episodes, {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
