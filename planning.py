#!/usr/bin/env python3
"""
Guide Dog Robot Planning Tool (model-based simulation)

Controls:
    P     Toggle policy/manual control
    SPACE Pause/Resume
    R     Reset position
    N     Generate new path
    ESC   Exit
    Arrows Manual control (when policy disabled)
"""
import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame
import torch

# Allow importing project modules when running from the FollowDataset directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add diffusion_policy submodule to Python path
DIFFUSION_POLICY_DIR = PROJECT_ROOT / "diffusion_policy"
if str(DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_POLICY_DIR))

from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.visualizer import Visualizer
from src.scoring import TrajectoryScorer

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
    TrainDiffusionUnetLowdimWorkspace,
)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ModelPlanner:
    """Run simulation and control robot with a trained policy."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
        use_ema: bool = True,
        action_mode: Optional[str] = None,
        k_lookahead: Optional[int] = None,
        frame_stride: Optional[int] = None,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 0.5,
        fps: int = 50,
        inference_steps: int = 8,
    ):
        self.fps = fps
        self.base_dt = 1.0 / fps
        self.leash_length = leash_length

        # Initialize modules
        self.path_generator = PathGenerator(target_length=path_length)
        self.visualizer = Visualizer()

        # Load policy
        self.device = resolve_device(device)
        self.workspace = TrainDiffusionUnetLowdimWorkspace.create_from_checkpoint(
            str(checkpoint_path)
        )
        if use_ema and getattr(self.workspace, "ema_model", None) is not None:
            self.policy = self.workspace.ema_model
        else:
            self.policy = self.workspace.model
        self.policy.to(self.device)
        self.policy.eval()

        cfg_action_mode = None
        try:
            cfg_action_mode = self.workspace.cfg.task.dataset.get("action_mode")
        except Exception:
            cfg_action_mode = None
        self.action_mode = action_mode or cfg_action_mode or "delta"

        cfg_k_lookahead = None
        try:
            cfg_k_lookahead = self.workspace.cfg.task.get("k_lookahead")
        except Exception:
            cfg_k_lookahead = None
        if cfg_k_lookahead is None:
            try:
                cfg_k_lookahead = self.workspace.cfg.task.dataset.get("k_lookahead")
            except Exception:
                cfg_k_lookahead = None
        if cfg_k_lookahead is None:
            try:
                cfg_k_lookahead = self.workspace.cfg.task.get("lookahead_stride")
            except Exception:
                cfg_k_lookahead = None
        if cfg_k_lookahead is None:
            try:
                cfg_k_lookahead = self.workspace.cfg.task.dataset.get("lookahead_stride")
            except Exception:
                cfg_k_lookahead = None

        cfg_frame_stride = None
        try:
            cfg_frame_stride = self.workspace.cfg.task.get("frame_stride")
        except Exception:
            cfg_frame_stride = None
        if cfg_frame_stride is None:
            try:
                cfg_frame_stride = self.workspace.cfg.task.dataset.get("frame_stride")
            except Exception:
                cfg_frame_stride = None

        self.obs_dim = int(self.policy.obs_dim)
        self.action_dim = int(self.policy.action_dim)
        self.n_obs_steps = int(self.policy.n_obs_steps)
        self.n_action_steps = int(self.policy.n_action_steps)
        extra_obs = self.obs_dim - 4
        if extra_obs < 0 or extra_obs % 2 != 0:
            raise ValueError(f"Unsupported obs_dim={self.obs_dim}, expected 4 + 2*N")
        self.n_lookahead = extra_obs // 2
        stride = k_lookahead if k_lookahead is not None else cfg_k_lookahead
        if stride is None:
            stride = 5
        self.lookahead_stride = max(1, int(stride))

        stride = frame_stride if frame_stride is not None else cfg_frame_stride
        if stride is None:
            stride = 1
        self.frame_stride = max(1, int(stride))
        self.dt = self.base_dt * self.frame_stride
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.dt,
        )

        self.scorer = None
        self.current_path_data = None
        self.running = True
        self.paused = False
        self.use_policy = True

        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.lookahead_world = None
        self.frame_count = 0

        self.obs_history = deque(maxlen=self.n_obs_steps)
        
        # Performance optimization: cache actions and reduce inference frequency
        self.cached_action_seq = None
        self.cached_action_idx = 0
        self.inference_interval = max(1, self.n_action_steps // 2)  # Infer every N frames
        self.frames_since_inference = 0
        
        # Reduce inference steps for faster performance
        if hasattr(self.policy, 'num_inference_steps'):
            original_steps = self.policy.num_inference_steps
            self.policy.num_inference_steps = inference_steps
            if original_steps != inference_steps:
                print(f"Set inference steps to {inference_steps} (original: {original_steps}) for faster performance")

        self._generate_new_path()

    def _generate_new_path(self):
        """Generate new reference path."""
        self.current_path_data = self.path_generator.generate()
        self._reset_position()
        print(f"New path generated: length={self.current_path_data['length']:.1f}m")

    def _reset_position(self):
        """Reset to start position."""
        if self.current_path_data is not None:
            start = self.current_path_data["start"]
            self.scorer = TrajectoryScorer(
                self.current_path_data["path"],
                self.leash_length,
            )
        else:
            start = np.array([0.0, 0.0])

        self.physics.reset(start)
        self.robot_trajectory = []
        self.human_trajectory = []
        self.planned_path = None
        self.frame_count = 0
        self._seed_obs_history(self.physics.robot.position, self.physics.human.position)
        self.policy.reset()
        # Reset action cache
        self.cached_action_seq = None
        self.cached_action_idx = 0
        self.frames_since_inference = 0

    def _seed_obs_history(self, robot_pos: np.ndarray, human_pos: np.ndarray):
        obs = self._build_obs(robot_pos, human_pos, self.physics.robot.heading)
        self.obs_history.clear()
        for _ in range(self.n_obs_steps):
            self.obs_history.append(obs.copy())

    def _handle_input(self):
        """Handle keyboard input."""
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False

            self.visualizer.handle_zoom(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                elif event.key == pygame.K_n:
                    self._generate_new_path()
                elif event.key == pygame.K_p:
                    self.use_policy = not self.use_policy
                    mode = "policy" if self.use_policy else "manual"
                    print(f"Control mode: {mode}")

    def _get_manual_control(self) -> Tuple[float, float]:
        keys = pygame.key.get_pressed()
        forward = 0.0
        turn = 0.0

        if keys[pygame.K_UP]:
            forward = 1.0
        elif keys[pygame.K_DOWN]:
            forward = -1.0

        if keys[pygame.K_LEFT]:
            turn = 1.0
        elif keys[pygame.K_RIGHT]:
            turn = -1.0
        return forward, turn

    def _predict_action(self) -> np.ndarray:
        obs_seq = np.stack(self.obs_history, axis=0)
        obs_tensor = torch.from_numpy(obs_seq).to(
            device=self.device, dtype=self.policy.dtype
        )[None, ...]
        obs_dict = {"obs": obs_tensor}
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        action_seq = action_dict["action"].detach().cpu().numpy()[0]
        return action_seq.astype(np.float32)

    def _build_obs(
        self, robot_pos: np.ndarray, human_pos: np.ndarray, heading: float
    ) -> np.ndarray:
        base = np.concatenate([robot_pos, human_pos], axis=0).astype(np.float32)
        if self.n_lookahead <= 0 or self.current_path_data is None:
            self.lookahead_world = None
            return base
        ref_features = self._build_reference_features(robot_pos, heading)
        return np.concatenate([base, ref_features], axis=0).astype(np.float32)

    def _build_reference_features(self, robot_pos: np.ndarray, heading: float) -> np.ndarray:
        ref_path = self.current_path_data["path"]
        if ref_path is None or len(ref_path) == 0:
            self.lookahead_world = None
            return np.zeros((self.n_lookahead * 2,), dtype=np.float32)
        diffs = ref_path - robot_pos
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        indices = idx + np.arange(self.n_lookahead) * self.lookahead_stride
        indices = np.clip(indices, 0, len(ref_path) - 1)
        points = ref_path[indices]
        self.lookahead_world = points
        rel = points - robot_pos
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        local_x = cos_h * rel[:, 0] + sin_h * rel[:, 1]
        local_y = -sin_h * rel[:, 0] + cos_h * rel[:, 1]
        return np.stack([local_x, local_y], axis=-1).reshape(-1).astype(np.float32)

    def _actions_to_path(self, robot_pos: np.ndarray, action_seq: np.ndarray) -> Optional[np.ndarray]:
        if action_seq.size == 0:
            return None
        if self.action_mode == "delta":
            points = np.cumsum(
                np.vstack([robot_pos[None, :], action_seq]), axis=0
            )[1:]
        elif self.action_mode == "position":
            points = action_seq
        else:  # velocity
            points = robot_pos[None, :] + np.cumsum(action_seq * self.dt, axis=0)
        return points

    def _action_to_delta(self, action: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        if self.action_mode == "delta":
            return action
        if self.action_mode == "position":
            return action - robot_pos
        return action * self.dt

    def _delta_to_control(self, delta: np.ndarray, heading: float) -> Tuple[float, float]:
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            return 0.0, 0.0
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(desired_heading - heading)
        turn_speed = self.physics.turn_speed
        turn_input = heading_error / (turn_speed * self.dt) if turn_speed > 0 else 0.0
        turn_input = float(np.clip(turn_input, -1.0, 1.0))

        forward_input = delta_norm / (self.physics.robot_speed * self.dt)
        forward_input = float(np.clip(forward_input, -1.0, 1.0))
        return forward_input, turn_input

    def _step(self):
        """Advance simulation by one step."""
        forward = 0.0
        turn = 0.0

        if not self.paused:
            if self.use_policy:
                # Performance optimization: only infer every N frames
                if (self.cached_action_seq is None or 
                    self.cached_action_idx >= len(self.cached_action_seq) or
                    self.frames_since_inference >= self.inference_interval):
                    # Run inference
                    action_seq = self._predict_action()
                    self.cached_action_seq = action_seq
                    self.cached_action_idx = 0
                    self.frames_since_inference = 0
                    self.planned_path = self._actions_to_path(
                        self.physics.robot.position, action_seq
                    )
                else:
                    # Reuse cached actions
                    self.frames_since_inference += 1
                
                # Get current action from cached sequence
                if self.cached_action_seq is not None and len(self.cached_action_seq) > 0:
                    action = self.cached_action_seq[self.cached_action_idx]
                    self.cached_action_idx += 1
                    # Wrap around if needed
                    if self.cached_action_idx >= len(self.cached_action_seq):
                        self.cached_action_idx = len(self.cached_action_seq) - 1
                else:
                    action = np.zeros(self.action_dim)
                
                delta = self._action_to_delta(action, self.physics.robot.position)
                forward, turn = self._delta_to_control(delta, self.physics.robot.heading)
            else:
                forward, turn = self._get_manual_control()
                self.planned_path = None
                # Reset cache when switching to manual
                self.cached_action_seq = None
                self.cached_action_idx = 0
                self.frames_since_inference = 0

        self.physics.set_control(forward, turn)
        robot_state, human_state = self.physics.step()

        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())

        max_trail = 5000
        if len(self.robot_trajectory) > max_trail:
            self.robot_trajectory = self.robot_trajectory[-max_trail:]
            self.human_trajectory = self.human_trajectory[-max_trail:]

        if self.scorer:
            self.scorer.update(robot_state.position, human_state.position)

        obs = self._build_obs(
            robot_state.position, human_state.position, robot_state.heading
        )
        self.obs_history.append(obs)
        self.frame_count += 1
        return robot_state, human_state

    def _render(self, robot_state, human_state, actual_fps: float):
        scores = self.scorer.get_scores() if self.scorer else {}
        mode = "policy" if self.use_policy else "manual"
        if self.paused:
            mode += " (paused)"

        info = {
            "fps": actual_fps,
            "path_length": self.current_path_data["length"] if self.current_path_data else 0,
            "robot_x": robot_state.position[0],
            "robot_y": robot_state.position[1],
            "num_points": self.frame_count,
            "recording": False,
            "scores": scores,
            "mode": mode,
            "controls": [
                "P: Policy/Manual",
                "SPACE: Pause",
                "R: Reset",
                "N: New Path",
                "Arrows: Manual control",
                "Scroll: Zoom",
                "ESC: Exit",
            ],
        }

        self.visualizer.render(
            robot_pos=robot_state.position,
            robot_heading=robot_state.heading,
            human_pos=human_state.position,
            reference_path=self.current_path_data["path"] if self.current_path_data else None,
            robot_trajectory=self.robot_trajectory,
            human_trajectory=self.human_trajectory,
            planned_path=self.planned_path,
            lookahead_points=self.lookahead_world,
            start_pos=self.current_path_data["start"] if self.current_path_data else None,
            end_pos=self.current_path_data["end"] if self.current_path_data else None,
            leash_tension=self.physics.get_leash_tension(),
            info=info,
        )

    def run(self):
        print("=" * 60)
        print("Guide Dog Robot Planning Tool")
        print("=" * 60)
        print("Controls: P=Policy/Manual | SPACE=Pause | R=Reset | N=NewPath | ESC=Exit")
        print("=" * 60)

        while self.running:
            self._handle_input()
            robot_state, human_state = self._step()
            actual_fps = self.visualizer.tick(self.fps)
            self._render(robot_state, human_state, actual_fps)

        self.visualizer.quit()
        print("Program exit")


def main():
    parser = argparse.ArgumentParser(description="Guide-follow planning with a trained policy")
    
    # Default checkpoint path - use latest available checkpoint
    default_ckpt = Path(
        "/home/yyf/IROS2026/diffusion_policy/data/outputs/2026.01.13/20.17.23_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/latest.ckpt"
    )
    
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=default_ckpt,
        help="Path to the trained checkpoint (.ckpt)",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, or auto")
    parser.add_argument("--no-ema", action="store_true", help="Use non-EMA model")
    parser.add_argument("--action-mode", default=None, help="delta | position | velocity")
    parser.add_argument(
        "--k-lookahead",
        "--lookahead-stride",
        dest="k_lookahead",
        type=int,
        default=None,
        help="Sample one lookahead point every k path points (e.g., 5)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Downsample factor for simulated steps (matches dataset frame_stride)",
    )
    parser.add_argument("--path-length", type=float, default=50.0)
    parser.add_argument("--leash-length", type=float, default=1.5)
    parser.add_argument("--robot-speed", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=64, 
                        help="Number of diffusion inference steps (lower=faster, default=16)")
    args = parser.parse_args()

    # Validate checkpoint file exists
    if not args.ckpt.exists():
        print(f"Error: Checkpoint file not found: {args.ckpt}")
        print("\nAvailable checkpoints:")
        outputs_dir = Path("/home/yyf/IROS2026/diffusion_policy/data/outputs")
        if outputs_dir.exists():
            for ckpt in sorted(outputs_dir.rglob("*.ckpt")):
                print(f"  {ckpt}")
        sys.exit(1)

    planner = ModelPlanner(
        checkpoint_path=args.ckpt,
        device=args.device,
        use_ema=not args.no_ema,
        action_mode=args.action_mode,
        k_lookahead=args.k_lookahead,
        frame_stride=args.frame_stride,
        path_length=args.path_length,
        leash_length=args.leash_length,
        robot_speed=args.robot_speed,
        fps=args.fps,
        inference_steps=args.inference_steps,
    )
    planner.run()


if __name__ == "__main__":
    main()
