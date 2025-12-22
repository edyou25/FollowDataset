#!/usr/bin/env python3
"""
Dataset Replay - Playback and manage recorded trajectories

Controls:
    UP/DOWN    Select episode
    ENTER      Load selected episode
    BACKSPACE  Delete selected episode
    SPACE      Play/Pause
    LEFT/RIGHT Step backward/forward (when paused)
    R          Restart from beginning
    +/-        Speed up/down
    WASD       Pan camera
    Scroll     Zoom in/out
    ESC        Exit / Back
"""
import os
import sys
import shutil
import numpy as np
import pygame

from src.data_storage import DataStorage
from src.visualizer import Visualizer


class DatasetReplay:
    """Replay and manage recorded trajectories"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        self.visualizer.camera_follow = False  # Manual camera control
        
        # Episode list
        self.episodes = DataStorage.list_episodes(data_dir)
        self.selected_idx = 0
        
        # Current loaded data
        self.current_data = None
        self.current_meta = None
        self.current_episode = None
        
        # Playback state
        self.playing = False
        self.frame_idx = 0
        self.playback_speed = 1.0
        self.accumulated_time = 0.0
        
        # State
        self.running = True
        self.in_selection = True  # Start in selection mode
        self.confirm_delete = False  # Delete confirmation state
        self.delete_target = None  # Episode to delete
        
    def _refresh_episodes(self):
        """Refresh episode list"""
        self.episodes = DataStorage.list_episodes(self.data_dir)
        if self.selected_idx >= len(self.episodes):
            self.selected_idx = max(0, len(self.episodes) - 1)
    
    def _delete_episode(self, episode_name: str):
        """Delete an episode"""
        episode_dir = os.path.join(self.data_dir, episode_name)
        try:
            shutil.rmtree(episode_dir)
            print(f"Deleted: {episode_name}")
            self._refresh_episodes()
            return True
        except Exception as e:
            print(f"Delete failed: {e}")
            return False
    
    def _load_episode(self, episode_name: str):
        """Load an episode for playback"""
        episode_dir = os.path.join(self.data_dir, episode_name)
        self.current_data = DataStorage.load_episode(episode_dir)
        self.current_meta = self.current_data['metadata']
        self.current_episode = episode_name
        self.frame_idx = 0
        self.playing = False
        self.accumulated_time = 0.0
        self.in_selection = False
        
        # Center camera on path
        ref_path = np.array(self.current_meta['reference_path'])
        center = np.mean(ref_path, axis=0)
        self.visualizer.camera_offset = center
        
        print(f"Loaded: {episode_name}")
        print(f"  Frames: {self.current_meta['num_frames']}, Duration: {self.current_meta['duration_seconds']:.1f}s")
        scores = self.current_meta.get('scores', {})
        if scores:
            print(f"  Score: {scores.get('total', 'N/A')}/100 (Grade: {scores.get('grade', 'N/A')})")
    
    def _handle_input(self):
        """Handle keyboard input"""
        for event in self.visualizer.get_events():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Handle zoom
            self.visualizer.handle_zoom(event)
            
            if event.type == pygame.KEYDOWN:
                # Delete confirmation mode
                if self.confirm_delete:
                    if event.key == pygame.K_y:
                        self._delete_episode(self.delete_target)
                        self.confirm_delete = False
                        self.delete_target = None
                    elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                        self.confirm_delete = False
                        self.delete_target = None
                        print("Delete cancelled")
                    continue
                
                if event.key == pygame.K_ESCAPE:
                    if self.in_selection:
                        self.running = False
                    else:
                        # Return to selection
                        self.in_selection = True
                        self.current_data = None
                        self.playing = False
                
                elif self.in_selection:
                    # Selection mode controls
                    if event.key == pygame.K_UP:
                        self.selected_idx = max(0, self.selected_idx - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected_idx = min(len(self.episodes) - 1, self.selected_idx + 1)
                    elif event.key == pygame.K_RETURN:
                        if self.episodes:
                            self._load_episode(self.episodes[self.selected_idx])
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if self.episodes:
                            self.confirm_delete = True
                            self.delete_target = self.episodes[self.selected_idx]
                            print(f"Delete '{self.delete_target}'? Press Y to confirm, N to cancel")
                
                else:
                    # Playback mode controls
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    
                    elif event.key == pygame.K_r:
                        self.frame_idx = 0
                        self.accumulated_time = 0.0
                    
                    elif event.key == pygame.K_LEFT:
                        if not self.playing:
                            self.frame_idx = max(0, self.frame_idx - 1)
                    
                    elif event.key == pygame.K_RIGHT:
                        if not self.playing:
                            max_frame = len(self.current_data['robot_path']) - 1
                            self.frame_idx = min(max_frame, self.frame_idx + 1)
                    
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.playback_speed = min(4.0, self.playback_speed * 1.5)
                        print(f"Speed: {self.playback_speed:.1f}x")
                    
                    elif event.key == pygame.K_MINUS:
                        self.playback_speed = max(0.25, self.playback_speed / 1.5)
                        print(f"Speed: {self.playback_speed:.1f}x")
                    
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        # Delete current episode
                        if self.current_episode:
                            self.confirm_delete = True
                            self.delete_target = self.current_episode
                            print(f"Delete '{self.delete_target}'? Press Y to confirm, N to cancel")
        
        # WASD for camera pan (when not in selection)
        if not self.in_selection and not self.confirm_delete:
            keys = pygame.key.get_pressed()
            pan_speed = 0.5
            if keys[pygame.K_w]:
                self.visualizer.camera_offset[1] += pan_speed
            if keys[pygame.K_s]:
                self.visualizer.camera_offset[1] -= pan_speed
            if keys[pygame.K_a]:
                self.visualizer.camera_offset[0] -= pan_speed
            if keys[pygame.K_d]:
                self.visualizer.camera_offset[0] += pan_speed
    
    def _update(self, dt: float):
        """Update playback state"""
        if not self.playing or self.current_data is None:
            return
        
        self.accumulated_time += dt * self.playback_speed
        
        # Find frame based on timestamp
        timestamps = self.current_data['timestamps']
        while self.frame_idx < len(timestamps) - 1:
            if timestamps[self.frame_idx + 1] <= self.accumulated_time:
                self.frame_idx += 1
            else:
                break
        
        # Check if reached end
        if self.frame_idx >= len(timestamps) - 1:
            self.playing = False
    
    def _render_selection(self):
        """Render episode selection screen"""
        self.visualizer.screen.fill((25, 25, 35))
        
        # Title
        title = self.visualizer.font_large.render("Select Episode", True, (220, 220, 220))
        self.visualizer.screen.blit(title, (self.visualizer.width // 2 - 80, 30))
        
        # Episode count
        count_text = self.visualizer.font.render(f"Total: {len(self.episodes)} episodes", True, (150, 150, 150))
        self.visualizer.screen.blit(count_text, (self.visualizer.width - 180, 35))
        
        if not self.episodes:
            no_data = self.visualizer.font.render("No episodes found in data/", True, (150, 150, 150))
            self.visualizer.screen.blit(no_data, (self.visualizer.width // 2 - 100, 100))
        else:
            # Episode list
            y = 80
            for i, ep in enumerate(self.episodes):
                # Load metadata for display
                try:
                    meta_path = os.path.join(self.data_dir, ep, "metadata.json")
                    import json
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    scores = meta.get('scores', {})
                    grade = scores.get('grade', '?')
                    total = scores.get('total', 0)
                    frames = meta.get('num_frames', 0)
                    duration = meta.get('duration_seconds', 0)
                    
                    # Format display
                    display_text = f"{ep}  |  {frames} frames  |  {duration:.1f}s  |  Score: {total:.0f} ({grade})"
                except:
                    display_text = ep
                
                # Highlight selected
                if i == self.selected_idx:
                    color = (100, 200, 255)
                    # Selection indicator
                    pygame.draw.rect(self.visualizer.screen, (40, 60, 80), 
                                   (40, y - 5, self.visualizer.width - 80, 30), border_radius=5)
                    indicator = self.visualizer.font.render(">", True, color)
                    self.visualizer.screen.blit(indicator, (50, y))
                else:
                    color = (180, 180, 180)
                
                text = self.visualizer.font.render(display_text, True, color)
                self.visualizer.screen.blit(text, (70, y))
                y += 35
        
        # Delete confirmation overlay
        if self.confirm_delete:
            # Dim background
            overlay = pygame.Surface((self.visualizer.width, self.visualizer.height))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(180)
            self.visualizer.screen.blit(overlay, (0, 0))
            
            # Confirmation dialog
            dialog_w, dialog_h = 500, 150
            dialog_x = (self.visualizer.width - dialog_w) // 2
            dialog_y = (self.visualizer.height - dialog_h) // 2
            
            pygame.draw.rect(self.visualizer.screen, (50, 50, 60), 
                           (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=10)
            pygame.draw.rect(self.visualizer.screen, (100, 100, 120), 
                           (dialog_x, dialog_y, dialog_w, dialog_h), 2, border_radius=10)
            
            # Warning text
            warn_text = self.visualizer.font_large.render("Delete Episode?", True, (255, 100, 100))
            self.visualizer.screen.blit(warn_text, (dialog_x + dialog_w // 2 - 80, dialog_y + 20))
            
            ep_text = self.visualizer.font.render(self.delete_target, True, (200, 200, 200))
            self.visualizer.screen.blit(ep_text, (dialog_x + dialog_w // 2 - 100, dialog_y + 60))
            
            hint_text = self.visualizer.font.render("Press Y to confirm, N to cancel", True, (150, 150, 150))
            self.visualizer.screen.blit(hint_text, (dialog_x + dialog_w // 2 - 130, dialog_y + 100))
        
        # Controls hint
        hints = "UP/DOWN: Select  |  ENTER: Load  |  BACKSPACE: Delete  |  ESC: Exit"
        hint_surface = self.visualizer.font.render(hints, True, (100, 100, 100))
        self.visualizer.screen.blit(hint_surface, (self.visualizer.width // 2 - 230, self.visualizer.height - 40))
        
        # Single flip at the end
        pygame.display.flip()
    
    def _render_playback(self):
        """Render playback screen"""
        if self.current_data is None:
            return
        
        robot_path = self.current_data['robot_path']
        human_path = self.current_data['human_path']
        timestamps = self.current_data['timestamps']
        ref_path = np.array(self.current_meta['reference_path'])
        start_pos = np.array(self.current_meta['start_position'])
        end_pos = np.array(self.current_meta['end_position'])
        
        # Current positions
        robot_pos = robot_path[self.frame_idx]
        human_pos = human_path[self.frame_idx]
        
        # Calculate heading from trajectory
        if self.frame_idx > 0:
            diff = robot_pos - robot_path[self.frame_idx - 1]
            heading = np.arctan2(diff[1], diff[0])
        else:
            heading = 0.0
        
        # Leash tension
        leash_dist = np.linalg.norm(robot_pos - human_pos)
        leash_tension = min(1.0, leash_dist / 1.5)
        
        # Trajectory up to current frame
        robot_trajectory = robot_path[:self.frame_idx + 1]
        human_trajectory = human_path[:self.frame_idx + 1]
        
        # Scores
        scores = self.current_meta.get('scores', {})
        
        # Info
        info = {
            'fps': self.visualizer.clock.get_fps(),
            'path_length': self.current_meta.get('reference_path_length', 0),
            'robot_x': robot_pos[0],
            'robot_y': robot_pos[1],
            'num_points': len(robot_path),
            'recording': False,
            'scores': scores,
        }
        
        # Render (no flip, we'll flip after overlay)
        self.visualizer.render(
            robot_pos=robot_pos,
            robot_heading=heading,
            human_pos=human_pos,
            reference_path=ref_path,
            robot_trajectory=robot_trajectory,
            human_trajectory=human_trajectory,
            start_pos=start_pos,
            end_pos=end_pos,
            leash_tension=leash_tension,
            info=info,
            flip=False
        )
        
        # Playback overlay (no flip inside)
        self._draw_playback_overlay()
        
        # Delete confirmation overlay
        if self.confirm_delete:
            self._draw_delete_dialog()
        
        # Single flip at the end
        pygame.display.flip()
    
    def _draw_delete_dialog(self):
        """Draw delete confirmation dialog (no flip)"""
        # Dim background
        overlay = pygame.Surface((self.visualizer.width, self.visualizer.height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.visualizer.screen.blit(overlay, (0, 0))
        
        # Confirmation dialog
        dialog_w, dialog_h = 500, 150
        dialog_x = (self.visualizer.width - dialog_w) // 2
        dialog_y = (self.visualizer.height - dialog_h) // 2
        
        pygame.draw.rect(self.visualizer.screen, (50, 50, 60), 
                       (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=10)
        pygame.draw.rect(self.visualizer.screen, (100, 100, 120), 
                       (dialog_x, dialog_y, dialog_w, dialog_h), 2, border_radius=10)
        
        # Warning text
        warn_text = self.visualizer.font_large.render("Delete Episode?", True, (255, 100, 100))
        self.visualizer.screen.blit(warn_text, (dialog_x + dialog_w // 2 - 80, dialog_y + 20))
        
        ep_text = self.visualizer.font.render(self.delete_target, True, (200, 200, 200))
        self.visualizer.screen.blit(ep_text, (dialog_x + dialog_w // 2 - 100, dialog_y + 60))
        
        hint_text = self.visualizer.font.render("Press Y to confirm, N to cancel", True, (150, 150, 150))
        self.visualizer.screen.blit(hint_text, (dialog_x + dialog_w // 2 - 130, dialog_y + 100))
    
    def _draw_playback_overlay(self):
        """Draw playback controls overlay (no flip)"""
        # Progress bar
        bar_width = 400
        bar_height = 8
        bar_x = (self.visualizer.width - bar_width) // 2
        bar_y = self.visualizer.height - 60
        
        # Background
        pygame.draw.rect(self.visualizer.screen, (60, 60, 70), 
                        (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Progress
        if self.current_data:
            progress = self.frame_idx / max(1, len(self.current_data['robot_path']) - 1)
            pygame.draw.rect(self.visualizer.screen, (100, 180, 255), 
                            (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=4)
        
        # Time display
        if self.current_data:
            current_time = self.current_data['timestamps'][self.frame_idx]
            total_time = self.current_meta['duration_seconds']
            time_text = f"{current_time:.1f}s / {total_time:.1f}s"
        else:
            time_text = "0.0s / 0.0s"
        
        time_surface = self.visualizer.font.render(time_text, True, (200, 200, 200))
        self.visualizer.screen.blit(time_surface, (bar_x + bar_width + 15, bar_y - 5))
        
        # Play/Pause indicator
        status = "Playing" if self.playing else "Paused"
        status_color = (100, 255, 100) if self.playing else (255, 200, 100)
        status_surface = self.visualizer.font.render(status, True, status_color)
        self.visualizer.screen.blit(status_surface, (bar_x - 80, bar_y - 5))
        
        # Speed indicator
        speed_text = f"{self.playback_speed:.1f}x"
        speed_surface = self.visualizer.font.render(speed_text, True, (150, 150, 150))
        self.visualizer.screen.blit(speed_surface, (bar_x + bar_width // 2 - 20, bar_y + 15))
        
        # Frame counter
        if self.current_data:
            frame_text = f"Frame: {self.frame_idx + 1}/{len(self.current_data['robot_path'])}"
            frame_surface = self.visualizer.font.render(frame_text, True, (150, 150, 150))
            self.visualizer.screen.blit(frame_surface, (bar_x, bar_y + 15))
        
        # Episode name
        if self.current_episode:
            ep_surface = self.visualizer.font.render(self.current_episode, True, (180, 180, 180))
            self.visualizer.screen.blit(ep_surface, (self.visualizer.width // 2 - 80, 15))
        
        # Controls hint
        hint = "SPACE: Play/Pause | LEFT/RIGHT: Step | R: Restart | +/-: Speed | BACKSPACE: Delete | ESC: Back"
        hint_surface = self.visualizer.font.render(hint, True, (100, 100, 100))
        self.visualizer.screen.blit(hint_surface, (self.visualizer.width // 2 - 320, self.visualizer.height - 25))
    
    def run(self):
        """Main loop"""
        print("=" * 50)
        print("Dataset Replay")
        print("=" * 50)
        
        clock = pygame.time.Clock()
        fps = 60
        
        while self.running:
            dt = clock.tick(fps) / 1000.0
            
            self._handle_input()
            
            if self.in_selection:
                self._render_selection()
            else:
                self._update(dt)
                self._render_playback()
        
        self.visualizer.quit()
        print("Replay exit")


def main():
    """Entry point"""
    replay = DatasetReplay(data_dir="data")
    replay.run()


if __name__ == "__main__":
    main()

