#!/usr/bin/env python3
"""
Compare two guide-following policies in the same simulated environment.

Controls:
    SPACE   Pause/Resume (both)
    R       Reset episode (both)
    N       New shared path (both)
    P       Toggle policy/manual (both)
    1       Toggle policy/manual (A)
    2       Toggle policy/manual (B)
    C       Cycle camera target (avg/A/B)
    ESC     Exit
    Scroll  Zoom
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

# Ensure local imports work when running from repo root.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning import ModelPlanner
from src.path_generator import PathGenerator
from src.visualizer import Visualizer


def _resolve_checkpoint(path: Path) -> Optional[Path]:
    path = Path(path)
    return path if path.exists() else None


def _default_name_from_ckpt(ckpt: Optional[Path], fallback: str) -> str:
    if ckpt is None:
        return fallback
    stem = ckpt.stem
    if stem == "latest":
        # Use parent folder name for readability.
        return ckpt.parent.parent.name if ckpt.parent.parent else fallback
    return stem


def _draw_robot(
    viz: Visualizer, position: np.ndarray, heading: float, color: Tuple[int, int, int]
):
    screen_pos = viz.world_to_screen(position)
    size = 12
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    front = (
        screen_pos[0] + int(cos_h * size * 1.5),
        screen_pos[1] - int(sin_h * size * 1.5),
    )
    left = (
        screen_pos[0] + int(-cos_h * size - sin_h * size),
        screen_pos[1] - int(-sin_h * size + cos_h * size),
    )
    right = (
        screen_pos[0] + int(-cos_h * size + sin_h * size),
        screen_pos[1] - int(-sin_h * size - cos_h * size),
    )
    pygame.draw.polygon(viz.screen, color, [front, left, right])
    pygame.draw.polygon(viz.screen, (255, 255, 255), [front, left, right], 2)


def _draw_human(viz: Visualizer, position: np.ndarray, color: Tuple[int, int, int]):
    screen_pos = viz.world_to_screen(position)
    pygame.draw.circle(viz.screen, color, screen_pos, 10)
    pygame.draw.circle(viz.screen, (255, 255, 255), screen_pos, 10, 2)


def _draw_panel(
    viz: Visualizer,
    x: int,
    y: int,
    title: str,
    title_color: Tuple[int, int, int],
    lines: list[str],
    width: int = 390,
):
    pad = 10
    line_h = 22
    header_h = 30
    height = header_h + pad + len(lines) * line_h + pad
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(viz.screen, (35, 35, 50), rect, border_radius=10)
    pygame.draw.rect(viz.screen, (60, 60, 80), rect, 2, border_radius=10)
    title_surf = viz.font_large.render(title, True, title_color)
    viz.screen.blit(title_surf, (x + pad, y + 6))
    yy = y + header_h
    for line in lines:
        surf = viz.font.render(line, True, viz.COLORS["text"])
        viz.screen.blit(surf, (x + pad, yy))
        yy += line_h


def _format_score_line(planner: ModelPlanner) -> str:
    if planner.scorer is None:
        return "Score: N/A"
    scores = planner.scorer.get_scores()
    total = scores.get("total", 0.0)
    grade = scores.get("grade", "N/A")
    return f"Score: {total:.1f} ({grade})"


def _format_detail_lines(planner: ModelPlanner, name: str, device: str) -> list[str]:
    mode = "policy" if planner.use_policy else "manual"
    if getattr(planner, "collision_pause", False):
        mode += " (collision)"
    if planner.paused:
        mode += " (paused)"
    frame = int(planner.frame_count)
    return [
        f"Mode: {mode}",
        f"Frame: {frame}",
        f"Device: {device}",
        _format_score_line(planner),
        f"Collisions: {getattr(planner, '_compare_collisions', 0)}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Compare two policies in the same simulation")

    default_ckpt_a = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
    )
    default_ckpt_b = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/15.00.42_train_diffusion_transformer_lowdim_guide_guide_lowdim/checkpoints/epoch=0760-test_mean_score=0.580.ckpt"
    )
    parser.add_argument("--ckpt-a", type=Path, default=default_ckpt_a, help="Checkpoint for model A")
    parser.add_argument("--ckpt-b", type=Path, default=default_ckpt_b, help="Checkpoint for model B")
    parser.add_argument("--name-a", default=None, help="Display name for model A")
    parser.add_argument("--name-b", default=None, help="Display name for model B")

    parser.add_argument("--device-a", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--device-b", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--no-ema-a", action="store_true", help="Use non-EMA model for A")
    parser.add_argument("--no-ema-b", action="store_true", help="Use non-EMA model for B")

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--k-lookahead", type=int, default=None)
    parser.add_argument("--action-mode", default=None, help="delta | forward_heading | position | velocity")
    parser.add_argument("--path-length", type=float, default=50.0)
    parser.add_argument("--leash-length", type=float, default=1.5)
    parser.add_argument("--robot-speed", type=float, default=1.5)
    parser.add_argument("--inference-steps", type=int, default=64)

    parser.add_argument("--seed", type=int, default=None, help="Deterministic path generation seed")
    parser.add_argument(
        "--no-sync-reset",
        action="store_true",
        help="Do not reset both after both have collided (collided robot will stay paused until manual reset/new path)",
    )
    parser.add_argument(
        "--no-sync-pause",
        action="store_true",
        help="Do not pause both when one reaches goal",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="Directory for compare logs (jsonl)",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable logs")

    args = parser.parse_args()

    ckpt_a = _resolve_checkpoint(args.ckpt_a)
    ckpt_b = _resolve_checkpoint(args.ckpt_b)
    if ckpt_a is None:
        print(f"[warn] ckpt-a not found: {args.ckpt_a} (A will run manual)")
    if ckpt_b is None:
        print(f"[warn] ckpt-b not found: {args.ckpt_b} (B will run manual)")

    name_a = args.name_a or _default_name_from_ckpt(ckpt_a, "A")
    name_b = args.name_b or _default_name_from_ckpt(ckpt_b, "B")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_a = None if args.no_log else (log_dir / f"compare_{name_a}_{timestamp}.jsonl")
    log_b = None if args.no_log else (log_dir / f"compare_{name_b}_{timestamp}.jsonl")

    planner_a = ModelPlanner(
        checkpoint_path=ckpt_a,
        device=args.device_a,
        use_ema=not args.no_ema_a,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        log_path=log_a,
        create_visualizer=False,
        collision_behavior="pause",
    )
    planner_b = ModelPlanner(
        checkpoint_path=ckpt_b,
        device=args.device_b,
        use_ema=not args.no_ema_b,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        log_path=log_b,
        create_visualizer=False,
        collision_behavior="pause",
    )

    # Start in policy mode when checkpoints are available.
    planner_a.use_policy = planner_a.policy is not None
    planner_b.use_policy = planner_b.policy is not None

    # Shared path generator.
    path_generator = PathGenerator(target_length=args.path_length)
    path_counter = 0

    def apply_new_shared_path():
        nonlocal path_counter
        if args.seed is not None:
            np.random.seed(int(args.seed) + int(path_counter))
        path_data = path_generator.generate()
        planner_a.set_path_data(path_data, reset=True)
        planner_b.set_path_data(path_data, reset=True)
        path_counter += 1

    apply_new_shared_path()

    # One shared visualizer window.
    viz = Visualizer()
    running = True
    sync_reset = not args.no_sync_reset
    sync_pause = not args.no_sync_pause
    camera_mode = "avg"  # avg | a | b
    collided_a = False
    collided_b = False

    # Per-model collision counters (for quick comparison).
    setattr(planner_a, "_compare_collisions", 0)
    setattr(planner_b, "_compare_collisions", 0)

    print("=" * 70)
    print("Compare Planning")
    print(f"A: {name_a} | ckpt: {ckpt_a if ckpt_a else 'manual'} | device: {args.device_a}")
    print(f"B: {name_b} | ckpt: {ckpt_b if ckpt_b else 'manual'} | device: {args.device_b}")
    print("Controls: SPACE Pause | R Reset | N NewPath | P ToggleBoth | 1/2 Toggle | C Camera | ESC Exit")
    print("=" * 70)

    try:
        while running:
            for event in viz.get_events():
                if event.type == pygame.QUIT:
                    running = False
                    break

                viz.handle_event(event)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_SPACE:
                        paused = not bool(planner_a.paused and planner_b.paused)
                        planner_a.paused = paused
                        planner_b.paused = paused
                    elif event.key == pygame.K_r:
                        planner_a._reset_position()
                        planner_b._reset_position()
                        collided_a = False
                        collided_b = False
                    elif event.key == pygame.K_n:
                        apply_new_shared_path()
                        collided_a = False
                        collided_b = False
                    elif event.key == pygame.K_p:
                        if planner_a.policy is not None:
                            planner_a.use_policy = not bool(planner_a.use_policy)
                        if planner_b.policy is not None:
                            planner_b.use_policy = not bool(planner_b.use_policy)
                    elif event.key == pygame.K_1:
                        if planner_a.policy is not None:
                            planner_a.use_policy = not bool(planner_a.use_policy)
                    elif event.key == pygame.K_2:
                        if planner_b.policy is not None:
                            planner_b.use_policy = not bool(planner_b.use_policy)
                    elif event.key == pygame.K_c:
                        camera_mode = {"avg": "a", "a": "b", "b": "avg"}[camera_mode]

            if not running:
                break

            # Step both environments once.
            robot_a, human_a = planner_a._step()
            robot_b, human_b = planner_b._step()

            if planner_a.collision_happened:
                planner_a._compare_collisions += 1
                collided_a = True
            if planner_b.collision_happened:
                planner_b._compare_collisions += 1
                collided_b = True

            # One collision does NOT reset the other; only reset both once both have collided.
            if sync_reset and collided_a and collided_b:
                planner_a._reset_position()
                planner_b._reset_position()
                collided_a = False
                collided_b = False

            if sync_pause and (planner_a.paused or planner_b.paused):
                planner_a.paused = True
                planner_b.paused = True

            # Camera target.
            if camera_mode == "a":
                target = robot_a.position
            elif camera_mode == "b":
                target = robot_b.position
            else:
                target = 0.5 * (robot_a.position + robot_b.position)
            viz.update_camera(target)

            actual_fps = viz.tick(args.fps)

            # ----- Render -----
            viz.screen.fill(viz.COLORS["background"])
            if viz.layer_visibility.get("grid", True):
                viz.draw_grid()

            path_data = planner_a.current_path_data
            reference_path = path_data.get("path") if path_data else None
            obstacles = path_data.get("obstacles") if path_data else None
            segment_obstacles = path_data.get("segment_obstacles") if path_data else None
            start_pos = path_data.get("start") if path_data else None
            end_pos = path_data.get("end") if path_data else None

            if reference_path is not None and len(reference_path) > 0:
                if viz.layer_visibility.get("reference_path", True):
                    viz.draw_path(reference_path, viz.COLORS["path_ref"], 3)
            if obstacles is not None and len(obstacles) > 0:
                if viz.layer_visibility.get("obstacles", True):
                    viz.draw_obstacles(obstacles, (planner_a.physics.robot_radius, planner_a.physics.human_radius))
            if segment_obstacles is not None and len(segment_obstacles) > 0:
                if viz.layer_visibility.get("segment_obstacles", True):
                    viz.draw_segments(
                        segment_obstacles, (planner_a.physics.robot_radius, planner_a.physics.human_radius)
                    )

            if start_pos is not None and viz.layer_visibility.get("start_end", True):
                viz.draw_marker(start_pos, viz.COLORS["start"], 10, "Start")
            if end_pos is not None and viz.layer_visibility.get("start_end", True):
                viz.draw_marker(end_pos, viz.COLORS["end"], 10, "End")

            # Model A (warm colors)
            color_robot_a = (255, 180, 50)
            color_human_a = (150, 200, 255)
            color_plan_a = (220, 200, 80)
            color_trail_robot_a = (255, 100, 100)
            color_trail_human_a = (100, 150, 255)

            # Model B (cool/magenta)
            color_robot_b = (200, 120, 255)
            color_human_b = (120, 255, 200)
            color_plan_b = (120, 220, 255)
            color_trail_robot_b = (200, 120, 255)
            color_trail_human_b = (120, 255, 200)

            # Planned paths
            if planner_a.planned_path is not None and len(planner_a.planned_path) > 1:
                if viz.layer_visibility.get("planned_path", True):
                    viz.draw_path(planner_a.planned_path, color_plan_a, 2)
            if planner_b.planned_path is not None and len(planner_b.planned_path) > 1:
                if viz.layer_visibility.get("planned_path", True):
                    viz.draw_path(planner_b.planned_path, color_plan_b, 2)

            # Trajectories
            if planner_a.robot_trajectory is not None and len(planner_a.robot_trajectory) > 1:
                if viz.layer_visibility.get("robot_trajectory", True):
                    viz.draw_path(np.array(planner_a.robot_trajectory), color_trail_robot_a, 2)
            if planner_a.human_trajectory is not None and len(planner_a.human_trajectory) > 1:
                if viz.layer_visibility.get("human_trajectory", True):
                    viz.draw_path(np.array(planner_a.human_trajectory), color_trail_human_a, 2)
            if planner_b.robot_trajectory is not None and len(planner_b.robot_trajectory) > 1:
                if viz.layer_visibility.get("robot_trajectory", True):
                    viz.draw_path(np.array(planner_b.robot_trajectory), color_trail_robot_b, 2)
            if planner_b.human_trajectory is not None and len(planner_b.human_trajectory) > 1:
                if viz.layer_visibility.get("human_trajectory", True):
                    viz.draw_path(np.array(planner_b.human_trajectory), color_trail_human_b, 2)

            # Leashes + agents
            if viz.layer_visibility.get("leash", True):
                viz.draw_leash(robot_a.position, human_a.position, planner_a.physics.get_leash_tension())
                viz.draw_leash(robot_b.position, human_b.position, planner_b.physics.get_leash_tension())

            if viz.layer_visibility.get("agent_radii", True):
                viz.draw_radius(robot_a.position, planner_a.physics.robot_radius, (255, 220, 120))
                viz.draw_radius(human_a.position, planner_a.physics.human_radius, (180, 230, 255))
                viz.draw_radius(robot_b.position, planner_b.physics.robot_radius, (230, 200, 255))
                viz.draw_radius(human_b.position, planner_b.physics.human_radius, (160, 255, 220))

            _draw_human(viz, human_a.position, color_human_a)
            _draw_robot(viz, robot_a.position, robot_a.heading, color_robot_a)
            _draw_human(viz, human_b.position, color_human_b)
            _draw_robot(viz, robot_b.position, robot_b.heading, color_robot_b)

            # UI
            lines_a = _format_detail_lines(planner_a, name_a, args.device_a)
            lines_b = _format_detail_lines(planner_b, name_b, args.device_b)
            _draw_panel(viz, 15, 15, f"A: {name_a}", color_robot_a, lines_a)
            _draw_panel(viz, 15, 155, f"B: {name_b}", color_robot_b, lines_b)
            _draw_panel(
                viz,
                15,
                295,
                "Compare",
                (220, 220, 220),
                [
                    f"FPS: {actual_fps:.1f}",
                    f"Path length: {float(path_data.get('length', 0.0)):.1f}m" if path_data else "Path length: N/A",
                    f"Camera: {camera_mode}",
                    "SPACE Pause | R Reset | N NewPath | P Toggle | 1/2 | C Camera | ESC Exit",
                ],
                width=520,
            )

            viz.draw_legend()
            pygame.display.flip()

    finally:
        viz.quit()
        for planner in (planner_a, planner_b):
            if planner.log_fp is not None:
                try:
                    planner._log_event("shutdown")
                except Exception:
                    pass
                try:
                    planner.log_fp.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
