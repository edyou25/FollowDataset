#!/usr/bin/env python3
"""Evaluate collision reduction before/after QP safety filtering."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planning import ModelPlanner
from src.path_generator import PathGenerator


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _build_shared_paths(
    episodes: int,
    path_length: float,
    seed: int,
) -> list[dict[str, Any]]:
    generator = PathGenerator(target_length=path_length)
    bank = []
    for idx in range(int(episodes)):
        np.random.seed(int(seed) + idx)
        bank.append(copy.deepcopy(generator.generate()))
    return bank


def _run_episode(
    planner: ModelPlanner,
    path_data: dict[str, Any],
    max_frames: int,
    seed: int,
) -> dict[str, Any]:
    _set_all_seeds(int(seed))
    planner.set_path_data(path_data, reset=True)
    planner.use_policy = planner.policy is not None

    outcome = "timeout"
    collision_info = None
    for _ in range(int(max_frames)):
        planner._step()
        if planner.collision_happened:
            outcome = "collision"
            collision_info = copy.deepcopy(planner.collision_info)
            break
        if planner.paused and not planner.collision_pause:
            outcome = "success"
            break

    scores = planner.scorer.get_scores() if planner.scorer is not None else {}
    safety_stats = planner.episode_safety_stats
    return {
        "outcome": outcome,
        "frames": int(planner.frame_count),
        "score_total": float(scores.get("total", 0.0)),
        "score_grade": scores.get("grade", "N/A"),
        "safety_modified_steps": int(safety_stats.get("modified_steps", 0)),
        "safety_total_steps": int(safety_stats.get("total_steps", 0)),
        "safety_mean_shift": (
            float(safety_stats.get("total_shift", 0.0)) / max(1, int(safety_stats.get("total_steps", 0)))
        ),
        "safety_constraint_count": int(safety_stats.get("constraint_count", 0)),
        "safety_min_clearance": float(safety_stats.get("min_clearance", float("inf"))),
        "collision_who": collision_info.get("who") if collision_info else None,
        "collision_type": collision_info.get("type") if collision_info else None,
        "collision_idx": collision_info.get("idx") if collision_info else None,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(results))
    collisions = sum(1 for item in results if item["outcome"] == "collision")
    robot_collisions = sum(1 for item in results if item["collision_who"] == "robot")
    human_collisions = sum(1 for item in results if item["collision_who"] == "human")
    successes = sum(1 for item in results if item["outcome"] == "success")
    timeouts = sum(1 for item in results if item["outcome"] == "timeout")
    return {
        "episodes": len(results),
        "collisions": collisions,
        "robot_collisions": robot_collisions,
        "human_collisions": human_collisions,
        "collision_rate": collisions / total,
        "successes": successes,
        "success_rate": successes / total,
        "timeouts": timeouts,
        "timeout_rate": timeouts / total,
        "mean_score_total": float(np.mean([item["score_total"] for item in results])) if results else 0.0,
        "mean_frames": float(np.mean([item["frames"] for item in results])) if results else 0.0,
        "mean_safety_modified_steps": float(
            np.mean([item["safety_modified_steps"] for item in results])
        )
        if results
        else 0.0,
        "mean_safety_mean_shift": float(
            np.mean([item["safety_mean_shift"] for item in results])
        )
        if results
        else 0.0,
    }


def _evaluate_mode(
    mode_name: str,
    planner_kwargs: dict[str, Any],
    path_bank: list[dict[str, Any]],
    rollout_seed: int,
    max_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planner = ModelPlanner(**planner_kwargs)
    results = []
    try:
        for idx, path_data in enumerate(path_bank):
            episode_seed = int(rollout_seed) + idx
            result = _run_episode(planner, path_data, max_frames=max_frames, seed=episode_seed)
            result["episode"] = idx
            results.append(result)
            print(
                f"[{mode_name}] episode={idx:02d} outcome={result['outcome']:<9} "
                f"frames={result['frames']:<4d} score={result['score_total']:.1f} "
                f"modified={result['safety_modified_steps']}"
            )
    finally:
        if planner.log_fp is not None:
            planner.log_fp.close()
        del planner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, _summarize(results)


def main():
    default_ckpt = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.21/14.14.46_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0090-test_mean_score=0.630.ckpt"
    )

    parser = argparse.ArgumentParser(description="Compare diffusion vs robot/human QP safety filtering")
    parser.add_argument("--ckpt", type=Path, default=default_ckpt)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--path-length", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--k-lookahead", type=int, default=None)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--leash-length", type=float, default=1.5)
    parser.add_argument("--robot-speed", type=float, default=1.5)
    parser.add_argument("--inference-steps", type=int, default=64)
    parser.add_argument("--turn-gain", type=float, default=1.2)
    parser.add_argument("--curvature-scale", type=float, default=0.7)
    parser.add_argument("--min-speed-scale", type=float, default=0.25)
    parser.add_argument("--no-curvature-slowdown", action="store_true")
    parser.add_argument("--safety-margin", type=float, default=0.2)
    parser.add_argument("--safety-alpha", type=float, default=1.0)
    parser.add_argument("--safety-max-constraints", type=int, default=8)
    parser.add_argument("--safety-influence-distance", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to FollowDataset/logs/safety_eval_<timestamp>.json",
    )
    args = parser.parse_args()

    ckpt = _resolve_checkpoint(args.ckpt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (THIS_DIR / "logs" / f"safety_eval_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_bank = _build_shared_paths(
        episodes=args.episodes,
        path_length=args.path_length,
        seed=args.seed,
    )

    planner_common = dict(
        checkpoint_path=ckpt,
        device=args.device,
        use_ema=True,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
        turn_gain=args.turn_gain,
        curvature_slowdown=not args.no_curvature_slowdown,
        curvature_scale=args.curvature_scale,
        min_speed_scale=args.min_speed_scale,
        log_path=None,
        log_interval=1,
        create_visualizer=False,
        collision_behavior="pause",
    )

    print("=" * 72)
    print("Safety Evaluation")
    print(f"Checkpoint : {ckpt}")
    print(f"Episodes   : {args.episodes}")
    print(f"Path length: {args.path_length}")
    print(f"Max frames : {args.max_frames}")
    print(f"Device     : {args.device}")
    print("=" * 72)

    diffusion_results, diffusion_summary = _evaluate_mode(
        "diffusion",
        {
            **planner_common,
            "safety_mode": "off",
            "safety_margin": args.safety_margin,
            "safety_alpha": args.safety_alpha,
            "safety_max_constraints": args.safety_max_constraints,
            "safety_influence_distance": args.safety_influence_distance,
        },
        path_bank=path_bank,
        rollout_seed=args.seed * 1000,
        max_frames=args.max_frames,
    )

    robot_qp_results, robot_qp_summary = _evaluate_mode(
        "robot_qp",
        {
            **planner_common,
            "safety_mode": "robot_qp",
            "safety_margin": args.safety_margin,
            "safety_alpha": args.safety_alpha,
            "safety_max_constraints": args.safety_max_constraints,
            "safety_influence_distance": args.safety_influence_distance,
        },
        path_bank=path_bank,
        rollout_seed=args.seed * 1000,
        max_frames=args.max_frames,
    )

    human_robot_qp_results, human_robot_qp_summary = _evaluate_mode(
        "human_robot_qp",
        {
            **planner_common,
            "safety_mode": "human_robot_qp",
            "safety_margin": args.safety_margin,
            "safety_alpha": args.safety_alpha,
            "safety_max_constraints": args.safety_max_constraints,
            "safety_influence_distance": args.safety_influence_distance,
        },
        path_bank=path_bank,
        rollout_seed=args.seed * 1000,
        max_frames=args.max_frames,
    )

    comparison = {
        "robot_qp_vs_diffusion": {
            "collision_delta": int(robot_qp_summary["collisions"] - diffusion_summary["collisions"]),
            "robot_collision_delta": int(robot_qp_summary["robot_collisions"] - diffusion_summary["robot_collisions"]),
            "human_collision_delta": int(robot_qp_summary["human_collisions"] - diffusion_summary["human_collisions"]),
            "collision_rate_delta": float(robot_qp_summary["collision_rate"] - diffusion_summary["collision_rate"]),
            "success_rate_delta": float(robot_qp_summary["success_rate"] - diffusion_summary["success_rate"]),
            "mean_score_delta": float(robot_qp_summary["mean_score_total"] - diffusion_summary["mean_score_total"]),
        },
        "human_robot_qp_vs_diffusion": {
            "collision_delta": int(human_robot_qp_summary["collisions"] - diffusion_summary["collisions"]),
            "robot_collision_delta": int(human_robot_qp_summary["robot_collisions"] - diffusion_summary["robot_collisions"]),
            "human_collision_delta": int(human_robot_qp_summary["human_collisions"] - diffusion_summary["human_collisions"]),
            "collision_rate_delta": float(human_robot_qp_summary["collision_rate"] - diffusion_summary["collision_rate"]),
            "success_rate_delta": float(human_robot_qp_summary["success_rate"] - diffusion_summary["success_rate"]),
            "mean_score_delta": float(human_robot_qp_summary["mean_score_total"] - diffusion_summary["mean_score_total"]),
        },
        "human_robot_qp_vs_robot_qp": {
            "collision_delta": int(human_robot_qp_summary["collisions"] - robot_qp_summary["collisions"]),
            "robot_collision_delta": int(human_robot_qp_summary["robot_collisions"] - robot_qp_summary["robot_collisions"]),
            "human_collision_delta": int(human_robot_qp_summary["human_collisions"] - robot_qp_summary["human_collisions"]),
            "collision_rate_delta": float(human_robot_qp_summary["collision_rate"] - robot_qp_summary["collision_rate"]),
            "success_rate_delta": float(human_robot_qp_summary["success_rate"] - robot_qp_summary["success_rate"]),
            "mean_score_delta": float(human_robot_qp_summary["mean_score_total"] - robot_qp_summary["mean_score_total"]),
        },
    }

    payload = {
        "config": {
            "ckpt": str(ckpt),
            "episodes": int(args.episodes),
            "path_length": float(args.path_length),
            "max_frames": int(args.max_frames),
            "seed": int(args.seed),
            "device": args.device,
            "safety_margin": float(args.safety_margin),
            "safety_alpha": float(args.safety_alpha),
            "safety_max_constraints": int(args.safety_max_constraints),
            "safety_influence_distance": float(args.safety_influence_distance),
        },
        "diffusion": {
            "summary": diffusion_summary,
            "episodes": diffusion_results,
        },
        "robot_qp": {
            "summary": robot_qp_summary,
            "episodes": robot_qp_results,
        },
        "human_robot_qp": {
            "summary": human_robot_qp_summary,
            "episodes": human_robot_qp_results,
        },
        "comparison": comparison,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("-" * 72)
    print(
        "Diffusion        : "
        f"collisions={diffusion_summary['collisions']}/{diffusion_summary['episodes']} "
        f"(robot={diffusion_summary['robot_collisions']}, human={diffusion_summary['human_collisions']}), "
        f"score={diffusion_summary['mean_score_total']:.1f}"
    )
    print(
        "Robot QP         : "
        f"collisions={robot_qp_summary['collisions']}/{robot_qp_summary['episodes']} "
        f"(robot={robot_qp_summary['robot_collisions']}, human={robot_qp_summary['human_collisions']}), "
        f"score={robot_qp_summary['mean_score_total']:.1f}"
    )
    print(
        "Human+Robot QP   : "
        f"collisions={human_robot_qp_summary['collisions']}/{human_robot_qp_summary['episodes']} "
        f"(robot={human_robot_qp_summary['robot_collisions']}, human={human_robot_qp_summary['human_collisions']}), "
        f"score={human_robot_qp_summary['mean_score_total']:.1f}"
    )
    print(
        "Delta vs diffusion: "
        f"robot_qp={comparison['robot_qp_vs_diffusion']['collision_rate_delta']:+.1%}, "
        f"human_robot_qp={comparison['human_robot_qp_vs_diffusion']['collision_rate_delta']:+.1%}"
    )
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
