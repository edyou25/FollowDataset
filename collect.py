#!/usr/bin/env python3
"""
Guide Dog Robot Data Collection Tool

Controls:
    ↑/↓  Forward/Backward
    ←/→  Turn Left/Right
    SPACE Start/Stop Recording
    S     Save trajectory
    R     Reset position
    N     Generate new path
    ESC   Exit
"""
import sys
import os
import numpy as np
import pygame

from src.path_generator import PathGenerator
from src.physics import PhysicsEngine
from src.visualizer import Visualizer
from src.data_storage import DataStorage
from src.scoring import TrajectoryScorer


class DataCollector:
    """Main data collection class"""
    
    def __init__(
        self,
        path_length: float = 50.0,
        leash_length: float = 1.5,
        robot_speed: float = 2.0,
        fps: int = 60
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.leash_length = leash_length
        
        # Initialize modules
        self.path_generator = PathGenerator(target_length=path_length)
        self.physics = PhysicsEngine(
            leash_length=leash_length,
            robot_speed=robot_speed,
            dt=self.dt
        )
        self.visualizer = Visualizer()
        
        # Set data storage path to 'data' folder in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        self.storage = DataStorage(base_dir=data_dir)
        
        self.scorer = None  # Will be initialized with path
        
        # State
        self.running = True
        self.recording = False
        self.current_path_data = None
        
        # Trajectory cache (for visualization)
        self.robot_trajectory = []
        self.human_trajectory = []
        
        # Generate initial path
        self._generate_new_path()
    
    def _generate_new_path(self):
        """Generate new reference path"""
        self.current_path_data = self.path_generator.generate()
        self._reset_position()
        print(f"New path generated: length={self.current_path_data['length']:.1f}m")
    
    def _reset_position(self):
        """Reset to start position"""
        if self.current_path_data is not None:
            start = self.current_path_data['start']
            # Initialize scorer with new path
            self.scorer = TrajectoryScorer(
                self.current_path_data['path'],
                self.leash_length
            )
        else:
            start = np.array([0.0, 0.0])
        
        self.physics.reset(start)
        self.robot_trajectory = []
        self.human_trajectory = []
        
        if self.recording:
            self._stop_recording()
    
    def _start_recording(self):
        """Start recording"""
        self.recording = True
        self.storage.start_recording()
        self.robot_trajectory = []
        self.human_trajectory = []
        # Reset scorer for new recording
        if self.scorer:
            self.scorer.reset()
        print("Recording started...")
    
    def _stop_recording(self):
        """Stop recording"""
        self.recording = False
        # Show score summary
        if self.scorer:
            scores = self.scorer.get_scores()
            print(f"Recording stopped. Points: {self.storage.get_num_points()}")
            print(f"  Score: {scores['total']:.0f}/100 (Grade: {scores['grade']})")
    
    def _save_episode(self):
        """Save current trajectory"""
        if self.storage.get_num_points() == 0:
            print("No data to save!")
            return
        
        # Get final scores
        scores = self.scorer.get_scores() if self.scorer else {}
        
        # Check quality threshold
        if scores.get('total', 0) < 50:
            print(f"⚠ Low quality score: {scores.get('total', 0):.0f}/100 (Grade: {scores.get('grade', 'F')})")
            print("  Consider discarding this trajectory (press S again to force save)")
        
        try:
            episode_dir = self.storage.save_episode(
                reference_path=self.current_path_data['path'],
                start_pos=self.current_path_data['start'],
                end_pos=self.current_path_data['end'],
                obstacles=self.current_path_data.get('obstacles'),
                segment_obstacles=self.current_path_data.get('segment_obstacles'),
                extra_metadata={'scores': scores}
            )
            print(f"✓ Saved! Score: {scores.get('total', 0):.0f}/100 ({scores.get('grade', 'N/A')})")
            
            # Clear recording state
            self.recording = False
            self.storage.clear()
            self.robot_trajectory = []
            self.human_trajectory = []
            if self.scorer:
                self.scorer.reset()
        except Exception as e:
            print(f"Save failed: {e}")
    
    def _handle_input(self):
        """Handle keyboard input"""
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Handle mouse wheel zoom
            self.visualizer.handle_zoom(event)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_SPACE:
                    if self.recording:
                        self._stop_recording()
                    else:
                        self._start_recording()
                
                elif event.key == pygame.K_s:
                    self._save_episode()
                
                elif event.key == pygame.K_r:
                    self._reset_position()
                    print("Position reset")
                
                elif event.key == pygame.K_n:
                    self._generate_new_path()
        
        # Continuous key detection
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
        
        self.physics.set_control(forward, turn)
    
    def _update(self):
        """Update physics state"""
        robot_state, human_state = self.physics.step()

        if self._check_collision():
            return self.physics.robot.copy(), self.physics.human.copy()
        
        # Update trajectory cache
        self.robot_trajectory.append(robot_state.position.copy())
        self.human_trajectory.append(human_state.position.copy())
        
        # Limit trajectory length (avoid memory issues)
        max_trail = 5000
        if len(self.robot_trajectory) > max_trail:
            self.robot_trajectory = self.robot_trajectory[-max_trail:]
            self.human_trajectory = self.human_trajectory[-max_trail:]
        
        # Record data and update scorer
        if self.recording:
            self.storage.record_frame(robot_state.position, human_state.position)
            if self.scorer:
                self.scorer.update(robot_state.position, human_state.position)
        
        return robot_state, human_state

    def _check_collision(self) -> bool:
        obstacles = self.current_path_data.get('obstacles') if self.current_path_data else None
        segments = self.current_path_data.get('segment_obstacles') if self.current_path_data else None
        collided, info = self.physics.check_collision(obstacles, segment_obstacles=segments)
        if collided:
            who = info.get("who", "agent")
            idx = info.get("idx")
            obs_type = info.get("type", "obstacle")
            print(f"Collision detected ({who}, {obs_type} {idx}), resetting.")
            self._reset_position()
            return True
        return False
    
    def _render(self, robot_state, human_state, actual_fps: float):
        """Render frame"""
        # Get current scores
        scores = self.scorer.get_scores() if self.scorer else {}
        
        info = {
            'fps': actual_fps,
            'path_length': self.current_path_data['length'] if self.current_path_data else 0,
            'robot_x': robot_state.position[0],
            'robot_y': robot_state.position[1],
            'num_points': self.storage.get_num_points(),
            'recording': self.recording,
            'scores': scores,
            'robot_radius': self.physics.robot_radius,
            'human_radius': self.physics.human_radius,
        }
        
        self.visualizer.render(
            robot_pos=robot_state.position,
            robot_heading=robot_state.heading,
            human_pos=human_state.position,
            reference_path=self.current_path_data['path'] if self.current_path_data else None,
            robot_trajectory=self.robot_trajectory,
            human_trajectory=self.human_trajectory,
            obstacles=self.current_path_data.get('obstacles') if self.current_path_data else None,
            segment_obstacles=self.current_path_data.get('segment_obstacles') if self.current_path_data else None,
            obstacle_inflation=(self.physics.robot_radius, self.physics.human_radius),
            robot_radius=self.physics.robot_radius,
            human_radius=self.physics.human_radius,
            start_pos=self.current_path_data['start'] if self.current_path_data else None,
            end_pos=self.current_path_data['end'] if self.current_path_data else None,
            leash_tension=self.physics.get_leash_tension(),
            info=info
        )
    
    def run(self):
        """Main loop"""
        print("=" * 50)
        print("Guide Dog Robot Data Collection Tool")
        print("=" * 50)
        print("Controls: Arrows=Move | SPACE=Record | S=Save | R=Reset | N=NewPath | ESC=Exit")
        print("=" * 50)
        
        while self.running:
            # Handle input
            self._handle_input()
            
            # Update physics
            robot_state, human_state = self._update()
            
            # Control frame rate and get actual FPS
            actual_fps = self.visualizer.tick(self.fps)
            
            # Render
            self._render(robot_state, human_state, actual_fps)
        
        # Cleanup
        self.visualizer.quit()
        print("Program exit")


def main():
    """Entry point"""
    collector = DataCollector(
        path_length=50.0,   # Reference path length 50m
        leash_length=1.5,   # Leash length 1.5m
        robot_speed=1.5,    # Robot speed 1.5m/s (正常步行速度，匹配训练配置)
        fps=20              # Frame rate 20FPS (降低以增加每步时间，匹配训练配置)
    )
    collector.run()


if __name__ == "__main__":
    main()
