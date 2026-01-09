#!/usr/bin/env python3
"""Interactive Plotly visualization for a recorded episode (with a time slider)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow importing local modules when running from the data/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_storage import DataStorage
from src.scoring import TrajectoryScorer


def select_episode(data_dir: Path, target: str | None) -> str:
    """Pick an episode name (defaults to the latest one)."""
    episodes = DataStorage.list_episodes(str(data_dir))
    if not episodes:
        raise SystemExit(f"No episodes found in {data_dir}")
    if target:
        if target not in episodes:
            raise SystemExit(
                f"Episode '{target}' not found in {data_dir}. "
                f"Available: {', '.join(episodes)}"
            )
        return target
    # Names are timestamped, so the last item is usually the latest recording
    return episodes[-1]


def compute_timeseries(
    robot_path: np.ndarray, human_path: np.ndarray, reference_path: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Reuse the scorer to get deviation/tension series and summary scores."""
    scorer = TrajectoryScorer(reference_path)
    for robot_pos, human_pos in zip(robot_path, human_path):
        scorer.update(robot_pos, human_pos)
    return (
        np.array(scorer.path_deviations),
        np.array(scorer.leash_tensions),
        scorer.get_scores(),
    )


def prepare_trace_data(
    data: dict,
    mask: Optional[np.ndarray],
    mask_debug: Optional[Dict[str, np.ndarray]],
) -> Dict[str, object]:
    meta = data["metadata"]
    ref_path = np.array(meta["reference_path"])
    robot_path = data["robot_path"]
    human_path = data["human_path"]
    timestamps = data["timestamps"]

    path_dev, leash_tension, computed_scores = compute_timeseries(
        robot_path, human_path, ref_path
    )
    scores = meta.get("scores") or computed_scores

    mask_arr = mask if mask is not None else np.zeros(len(robot_path), dtype=bool)
    debug = mask_debug or {}

    def masked(arr: np.ndarray, selector: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        sel = selector if selector is not None else np.zeros(len(arr), dtype=bool)
        return np.where(sel, arr[:, 0], np.nan), np.where(sel, arr[:, 1], np.nan)

    x_list = []
    y_list = []
    # 0 ref, 1 human, 2 robot
    x_list.append(ref_path[:, 0])
    y_list.append(ref_path[:, 1])
    x_list.append(human_path[:, 0])
    y_list.append(human_path[:, 1])
    x_list.append(robot_path[:, 0])
    y_list.append(robot_path[:, 1])

    # 3 mask merged
    mx, my = masked(robot_path, mask_arr)
    x_list.append(mx)
    y_list.append(my)

    # 4-6 debug causes
    for key in ("cause_speed_jump", "cause_acc_event", "cause_heading_event"):
        dx, dy = masked(robot_path, debug.get(key))
        x_list.append(dx)
        y_list.append(dy)

    # 7 robot marker, 8 human marker
    x_list.append(np.array([robot_path[0, 0]]))
    y_list.append(np.array([robot_path[0, 1]]))
    x_list.append(np.array([human_path[0, 0]]))
    y_list.append(np.array([human_path[0, 1]]))

    # 9 start, 10 end
    x_list.append(np.array([meta["start_position"][0]]))
    y_list.append(np.array([meta["start_position"][1]]))
    x_list.append(np.array([meta["end_position"][0]]))
    y_list.append(np.array([meta["end_position"][1]]))

    # 11 path dev, 12 leash tension
    x_list.append(timestamps)
    y_list.append(path_dev)
    x_list.append(timestamps)
    y_list.append(leash_tension)

    indicator_ymin = float(min(path_dev.min(), leash_tension.min()))
    indicator_ymax = float(max(path_dev.max(), leash_tension.max()))
    if indicator_ymin == indicator_ymax:
        indicator_ymin -= 1.0
        indicator_ymax += 1.0
    # 13 indicator
    x_list.append(np.array([timestamps[0], timestamps[0]]))
    y_list.append(np.array([indicator_ymin, indicator_ymax]))

    score_text = ""
    if scores:
        score_text = (
            f"Total {scores.get('total', 'N/A')} "
            f"({scores.get('grade', 'N/A')}) · "
            f"Path {scores.get('path_following', 'N/A')} | "
            f"Smooth {scores.get('smoothness', 'N/A')} | "
            f"Complete {scores.get('completion', 'N/A')} | "
            f"Leash {scores.get('leash_control', 'N/A')}"
        )

    if len(timestamps) > 0:
        time_range = (float(timestamps[0]), float(timestamps[-1]))
    else:
        time_range = (0.0, 0.0)

    return {
        "meta": meta,
        "ref_path": ref_path,
        "robot_path": robot_path,
        "human_path": human_path,
        "timestamps": timestamps,
        "path_dev": path_dev,
        "leash_tension": leash_tension,
        "indicator_y": (indicator_ymin, indicator_ymax),
        "x_list": x_list,
        "y_list": y_list,
        "score_text": score_text,
        "time_range": time_range,
    }


def build_step_indices(n_frames: int, limit: int) -> List[int]:
    if n_frames <= 1:
        return [0]
    if limit and limit > 0 and n_frames > limit:
        stride = max(1, n_frames // limit)
    else:
        stride = 1
    indices = list(range(0, n_frames, stride))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)
    return indices


def build_frames_for_episode(
    trace_data: Dict[str, object],
    robot_marker_idx: int,
    human_marker_idx: int,
    indicator_trace_idx: int,
    max_frames: int,
    frame_prefix: str,
) -> Tuple[List[go.Frame], List[Dict[str, object]]]:
    timestamps = trace_data["timestamps"]
    robot_path = trace_data["robot_path"]
    human_path = trace_data["human_path"]
    indicator_ymin, indicator_ymax = trace_data["indicator_y"]

    step_indices = build_step_indices(len(timestamps), max_frames)
    frames: List[go.Frame] = []
    for idx in step_indices:
        frame_name = f"{frame_prefix}:{idx}"
        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(x=[robot_path[idx, 0]], y=[robot_path[idx, 1]]),
                    go.Scatter(x=[human_path[idx, 0]], y=[human_path[idx, 1]]),
                    go.Scatter(
                        x=[timestamps[idx], timestamps[idx]],
                        y=[indicator_ymin, indicator_ymax],
                    ),
                ],
                traces=[robot_marker_idx, human_marker_idx, indicator_trace_idx],
            )
        )

    slider_steps = [
        {
            "label": f"{timestamps[idx]:.2f}s",
            "method": "animate",
            "args": [
                [f"{frame_prefix}:{idx}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                },
            ],
        }
        for idx in step_indices
    ]

    return frames, slider_steps


def load_mask(
    episode_dir: Path, mask_name: Optional[str], expected_len: int
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Load mask npz if available; align length to timestamps and pull debug causes."""
    if not mask_name:
        return None, None
    mask_path = episode_dir / mask_name
    if not mask_path.exists():
        return None, None
    try:
        data = np.load(mask_path)
        mask = data["mask"].astype(bool)
        debug = {}
        for key in ("cause_speed_jump", "cause_acc_event", "cause_heading_event"):
            if key in data:
                debug[key] = data[key].astype(bool)
    except Exception as e:
        print(f"[warn] failed to load mask {mask_path}: {e}")
        return None, None
    if len(mask) != expected_len:
        if len(mask) > expected_len:
            mask = mask[:expected_len]
        else:
            pad = np.zeros(expected_len - len(mask), dtype=bool)
            mask = np.concatenate([mask, pad])
        print(f"[info] mask length adjusted to match timestamps: {len(mask)}")
    for k, v in list(debug.items()):
        if len(v) != expected_len:
            v = v[:expected_len] if len(v) > expected_len else np.concatenate(
                [v, np.zeros(expected_len - len(v), dtype=bool)]
            )
            debug[k] = v
    return mask, (debug if debug else None)


def mask_path_trace(path: np.ndarray, mask: Optional[np.ndarray], name: str, color: str, width: int) -> go.Scatter:
    """Create overlay for masked segments on the XY plot (returns placeholder if empty)."""
    x = path[:, 0].astype(float)
    y = path[:, 1].astype(float)
    mask_arr = mask if mask is not None else np.zeros(len(path), dtype=bool)
    x_masked = np.where(mask_arr, x, np.nan)
    y_masked = np.where(mask_arr, y, np.nan)
    return go.Scatter(
        x=x_masked,
        y=y_masked,
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        showlegend=True,
    )


def make_debug_traces(path: np.ndarray, debug: Optional[Dict[str, np.ndarray]]) -> List[go.Scatter]:
    """Build per-cause overlays (always return three traces as placeholders)."""
    traces: List[go.Scatter] = []
    cause_styles = [
        ("cause_speed_jump", "Speed jump", "rgba(255,127,14,0.8)"),
        ("cause_acc_event", "Accel spike", "rgba(148,103,189,0.8)"),
        ("cause_heading_event", "Heading change", "rgba(23,190,207,0.8)"),
    ]
    for key, label, color in cause_styles:
        arr = debug.get(key) if debug else None
        traces.append(mask_path_trace(path, arr, label, color, 4))
    return traces


def build_figure(
    data: dict,
    episode_name: str,
    max_frames: int = 400,
    mask: Optional[np.ndarray] = None,
    mask_debug: Optional[Dict[str, np.ndarray]] = None,
    enable_slider: bool = True,
) -> go.Figure:
    """Build a two-panel Plotly figure."""
    trace_data = prepare_trace_data(data, mask, mask_debug)
    meta = trace_data["meta"]
    ref_path = trace_data["ref_path"]
    robot_path = trace_data["robot_path"]
    human_path = trace_data["human_path"]
    timestamps = trace_data["timestamps"]
    path_dev = trace_data["path_dev"]
    leash_tension = trace_data["leash_tension"]
    indicator_ymin, indicator_ymax = trace_data["indicator_y"]
    x_list = trace_data["x_list"]
    y_list = trace_data["y_list"]

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
        specs=[[{"type": "xy"}], [{"secondary_y": True}]],
    )

    # Trajectory view
    # Static trajectories
    fig.add_trace(
        go.Scatter(
            x=x_list[0],
            y=y_list[0],
            mode="lines",
            name="Reference path",
            line=dict(color="#7f7f7f", dash="dash"),
        ),
        row=1,
        col=1,
    )  # trace 0
    fig.add_trace(
        go.Scatter(
            x=x_list[1],
            y=y_list[1],
            mode="lines",
            name="Human path",
            line=dict(color="#ff7f0e"),
        ),
        row=1,
        col=1,
    )  # trace 1
    fig.add_trace(
        go.Scatter(
            x=x_list[2],
            y=y_list[2],
            mode="lines",
            name="Robot path",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )  # trace 2

    # Mask overlay on XY (highlight event segments) + debug layers (placeholders included)
    fig.add_trace(
        go.Scatter(
            x=x_list[3],
            y=y_list[3],
            mode="lines",
            name="Event segment",
            line=dict(color="rgba(255,165,0,0.9)", width=6),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    debug_styles = [
        ("Speed jump", "rgba(255,127,14,0.8)"),
        ("Accel spike", "rgba(148,103,189,0.8)"),
        ("Heading change", "rgba(23,190,207,0.8)"),
    ]
    for i, (label, color) in enumerate(debug_styles, start=4):
        fig.add_trace(
            go.Scatter(
                x=x_list[i],
                y=y_list[i],
                mode="lines",
                name=label,
                line=dict(color=color, width=4),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Current markers (animated)
    robot_marker_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x_list[7],
            y=y_list[7],
            mode="markers",
            name="Robot @ t",
            marker=dict(color="#1f77b4", size=10, symbol="circle"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    human_marker_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=x_list[8],
            y=y_list[8],
            mode="markers",
            name="Human @ t",
            marker=dict(color="#ff7f0e", size=10, symbol="square"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Start/end markers
    fig.add_trace(
        go.Scatter(
            x=x_list[9],
            y=y_list[9],
            mode="markers+text",
            text=["Start"],
            textposition="top center",
            name="Start",
            marker=dict(color="#2ca02c", size=10, symbol="circle"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_list[10],
            y=y_list[10],
            mode="markers+text",
            text=["End"],
            textposition="top center",
            name="End",
            marker=dict(color="#d62728", size=10, symbol="x"),
        ),
        row=1,
        col=1,
    )

    # Deviation and leash tension over time
    fig.add_trace(
        go.Scatter(
            x=x_list[11],
            y=y_list[11],
            mode="lines",
            name="Path deviation (m)",
            line=dict(color="#9467bd"),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )  # trace 7
    fig.add_trace(
        go.Scatter(
            x=x_list[12],
            y=y_list[12],
            mode="lines",
            name="Leash tension (ratio)",
            line=dict(color="#2ca02c"),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )  # trace 8

    # Progress indicator on time-series (kept on primary y-axis to avoid layout redraws)
    fig.add_trace(
        go.Scatter(
            x=x_list[13],
            y=y_list[13],
            mode="lines",
            name="t",
            line=dict(color="#d62728", width=2, dash="dot"),
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )  # trace 9 (animated)
    indicator_trace_idx = len(fig.data) - 1
    fig.update_layout(
        meta={
            "trace_indices": {
                "robot_marker": robot_marker_idx,
                "human_marker": human_marker_idx,
                "indicator": indicator_trace_idx,
            }
        }
    )

    fig.update_layout(
        # title=f"{episode_name} — frames: {meta.get('num_frames', 'N/A')}, "
        # f"duration: {meta.get('duration_seconds', 0):.1f}s",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=40, t=90, b=60),
        annotations=[
            dict(
                text=trace_data["score_text"],
                x=1.0,
                y=1.14,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                font=dict(size=12),
            )
        ],
    )

    # Keep equal aspect ratio for XY plot
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1, title="Y (m)")
    fig.update_xaxes(row=1, col=1, title="X (m)")

    fig.update_yaxes(title_text="Deviation (m)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Leash tension", row=2, col=1, secondary_y=True)
    # Fix time range so animation不会重算范围导致曲线消失
    if len(timestamps) > 1:
        fig.update_xaxes(
            title_text="Time (s)",
            range=[float(timestamps[0]), float(timestamps[-1])],
            row=2,
            col=1,
        )
    else:
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    if enable_slider:
        frames, slider_steps = build_frames_for_episode(
            trace_data,
            robot_marker_idx,
            human_marker_idx,
            indicator_trace_idx,
            max_frames,
            episode_name,
        )
        fig.frames = frames
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "t = ", "suffix": " s"},
                    "pad": {"t": 50},
                    "steps": slider_steps,
                }
            ],
        )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot an episode from data/ with Plotly"
    )
    parser.add_argument(
        "-e",
        "--episode",
        help="Episode directory name under data/ (defaults to the latest one)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional HTML output path. If omitted, fig.show() is used.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Maximum frames for the slider (downsamples if needed, 0 to use all).",
    )
    parser.add_argument(
        "--mask-name",
        default="event_mask_rule_v0.npz",
        help="Mask filename inside episode dir (set empty to disable).",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        default=True,
        help="Enable dropdown to switch episodes.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / "data"
    episodes = DataStorage.list_episodes(str(data_dir))
    if not episodes:
        raise SystemExit(f"No episodes found in {data_dir}")
    if args.episode:
        if args.episode not in episodes:
            raise SystemExit(f"Episode '{args.episode}' not found in {data_dir}")
        # Put selected episode first
        episodes = [args.episode] + [ep for ep in episodes if ep != args.episode]
    episode_name = episodes[-1] if not args.episode else episodes[0]

    # Prepare data for first episode
    first_dir = data_dir / episode_name
    first_data = DataStorage.load_episode(str(first_dir))
    first_mask, first_debug = load_mask(first_dir, args.mask_name, expected_len=len(first_data["timestamps"]))

    fig = build_figure(
        first_data,
        episode_name,
        max_frames=args.max_frames,
        mask=first_mask,
        mask_debug=first_debug,
        enable_slider=True,
    )

    if args.menu:
        trace_indices = (fig.layout.meta or {}).get("trace_indices", {})
        robot_marker_idx = trace_indices.get("robot_marker")
        human_marker_idx = trace_indices.get("human_marker")
        indicator_trace_idx = trace_indices.get("indicator")
        if None in (robot_marker_idx, human_marker_idx, indicator_trace_idx):
            raise SystemExit("Missing trace indices for slider animation.")

        payloads = []
        all_frames: List[go.Frame] = []
        slider_steps_map: Dict[str, List[Dict[str, object]]] = {}
        for ep in episodes:
            ep_dir = data_dir / ep
            ep_data = DataStorage.load_episode(str(ep_dir))
            ep_mask, ep_debug = load_mask(ep_dir, args.mask_name, expected_len=len(ep_data["timestamps"]))
            trace_data = prepare_trace_data(ep_data, ep_mask, ep_debug)
            payloads.append((ep, trace_data))
            frames, slider_steps = build_frames_for_episode(
                trace_data,
                robot_marker_idx,
                human_marker_idx,
                indicator_trace_idx,
                args.max_frames,
                ep,
            )
            all_frames.extend(frames)
            slider_steps_map[ep] = slider_steps

        fig.frames = all_frames
        if episode_name in slider_steps_map:
            fig.update_layout(
                sliders=[
                    {
                        "active": 0,
                        "currentvalue": {"prefix": "t = ", "suffix": " s"},
                        "pad": {"t": 50},
                        "steps": slider_steps_map[episode_name],
                    }
                ]
            )

        buttons = []
        for ep, payload in payloads:
            slider_steps = slider_steps_map.get(ep, [])
            buttons.append(
                dict(
                    label=ep,
                    method="update",
                    args=[
                        {"x": payload["x_list"], "y": payload["y_list"]},
                        {
                            "annotations": [
                                dict(
                                    text=payload["score_text"],
                                    x=1.0,
                                    y=1.14,
                                    xref="paper",
                                    yref="paper",
                                    showarrow=False,
                                    align="right",
                                    font=dict(size=12),
                                )
                            ],
                            "xaxis2": {"range": list(payload["time_range"])},
                            "sliders": [
                                {
                                    "active": 0,
                                    "currentvalue": {"prefix": "t = ", "suffix": " s"},
                                    "pad": {"t": 50},
                                    "steps": slider_steps,
                                }
                            ],
                        },
                    ],
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.18,
                    yanchor="top",
                )
            ]
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.output))
        print(f"Saved interactive plot to {args.output}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
