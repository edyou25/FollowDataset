"""
Visualization module - pygame real-time display
"""
import pygame
import numpy as np
from typing import Optional


class Visualizer:
    """Visualization interface for data collection"""
    
    # Color definitions
    COLORS = {
        'background': (25, 25, 35),
        'grid': (45, 45, 55),
        'path_ref': (80, 200, 120),      # Reference path - green
        'path_plan': (220, 200, 80),     # Planned path - yellow
        'lookahead': (255, 240, 120),    # Lookahead points - bright yellow
        'path_robot': (255, 100, 100),   # Robot trajectory - red
        'path_human': (100, 150, 255),   # Human trajectory - blue
        'robot': (255, 180, 50),         # Robot - orange
        'robot_radius': (255, 220, 120),
        'human': (150, 200, 255),        # Human - light blue
        'human_radius': (180, 230, 255),
        'leash': (200, 200, 200),        # Leash
        'start': (50, 255, 50),          # Start - bright green
        'end': (255, 50, 50),            # End - bright red
        'text': (220, 220, 220),
        'recording': (255, 80, 80),
        'obstacle': (80, 80, 100),
        'obstacle_inflated': (255, 200, 80),
        'obstacle_segment': (100, 100, 130),
    }
    
    # Zoom settings
    MIN_PPM = 3.0    # Minimum pixels per meter
    MAX_PPM = 200.0   # Maximum pixels per meter
    ZOOM_STEP = 1.2  # Zoom multiplier per scroll
    
    def __init__(
        self,
        width: int = 2400,  # 从 1200 增加到 1600
        height: int = 1600,  # 从 800 增加到 1000
        pixels_per_meter: float = 12.0
    ):
        self.width = width
        self.height = height
        self.ppm = pixels_per_meter  # Pixels per meter
        
        # Camera offset (world coordinates)
        self.camera_offset = np.array([0.0, 0.0])
        self.camera_follow = True  # Follow robot
        
        # pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Guide Dog Robot - Data Collection")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
    def world_to_screen(self, world_pos: np.ndarray) -> tuple:
        """Convert world coordinates to screen coordinates"""
        # Position relative to camera
        relative = world_pos - self.camera_offset
        
        # Convert to screen coordinates (y-axis flipped)
        screen_x = self.width / 2 + relative[0] * self.ppm
        screen_y = self.height / 2 - relative[1] * self.ppm
        
        return int(screen_x), int(screen_y)
    
    def update_camera(self, robot_position: np.ndarray):
        """Update camera position"""
        if self.camera_follow:
            # Smooth follow robot
            self.camera_offset = self.camera_offset * 0.95 + robot_position * 0.05
    
    def handle_zoom(self, event: pygame.event.Event):
        """Handle mouse wheel zoom"""
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll up - zoom in
                self.ppm = min(self.ppm * self.ZOOM_STEP, self.MAX_PPM)
            elif event.y < 0:  # Scroll down - zoom out
                self.ppm = max(self.ppm / self.ZOOM_STEP, self.MIN_PPM)
    
    def draw_grid(self):
        """Draw grid"""
        # Calculate visible range based on zoom
        grid_spacing = 5.0  # 5m grid
        
        for i in range(-20, 21):
            # Vertical lines
            x = i * grid_spacing
            start = self.world_to_screen(np.array([x, -100]))
            end = self.world_to_screen(np.array([x, 100]))
            pygame.draw.line(self.screen, self.COLORS['grid'], start, end, 1)
            
            # Horizontal lines
            y = i * grid_spacing
            start = self.world_to_screen(np.array([-100, y]))
            end = self.world_to_screen(np.array([100, y]))
            pygame.draw.line(self.screen, self.COLORS['grid'], start, end, 1)
    
    def draw_path(self, path: np.ndarray, color: tuple, width: int = 2):
        """Draw path"""
        if len(path) < 2:
            return
        
        points = [self.world_to_screen(p) for p in path]
        pygame.draw.lines(self.screen, color, False, points, width)

    def draw_points(self, points: np.ndarray, color: tuple, radius: int = 5):
        """Draw a set of points."""
        if points is None or len(points) == 0:
            return
        for point in points:
            screen_pos = self.world_to_screen(point)
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, max(1, radius - 2), 1)

    def _parse_inflation(self, inflation: Optional[tuple]):
        if inflation is None:
            return None, None
        inflate_values = None
        inflate_colors = None
        if isinstance(inflation, (list, tuple)):
            values = [v for v in inflation if v is not None]
            if len(values) == 2:
                inflate_values = [float(values[0]), float(values[1])]
                if abs(inflate_values[0] - inflate_values[1]) < 1e-6:
                    inflate_values = [inflate_values[0]]
            else:
                inflate_values = [float(v) for v in values]
        else:
            inflate_values = [float(inflation)]
        if inflate_values is not None and inflate_colors is None:
            inflate_colors = [self.COLORS['obstacle_inflated']] * len(inflate_values)
        return inflate_values, inflate_colors

    def draw_obstacles(self, obstacles: Optional[np.ndarray], inflation: Optional[tuple] = None):
        """Draw circular obstacles defined as (x, y, radius)."""
        if obstacles is None or len(obstacles) == 0:
            return
        inflate_values, inflate_colors = self._parse_inflation(inflation)
        for obs in obstacles:
            if isinstance(obs, dict):
                x = float(obs.get("x", 0.0))
                y = float(obs.get("y", 0.0))
                r = float(obs.get("r", 0.0))
            else:
                x = float(obs[0])
                y = float(obs[1])
                r = float(obs[2])
            screen_pos = self.world_to_screen(np.array([x, y]))
            radius_px = max(2, int(r * self.ppm))
            pygame.draw.circle(self.screen, self.COLORS['obstacle'], screen_pos, radius_px)
            pygame.draw.circle(self.screen, (200, 200, 220), screen_pos, radius_px, 1)
            if inflate_values:
                for idx, value in enumerate(inflate_values):
                    if value <= 0:
                        continue
                    inflated_px = max(2, int((r + value) * self.ppm))
                    color = inflate_colors[min(idx, len(inflate_colors) - 1)]
                    pygame.draw.circle(self.screen, color, screen_pos, inflated_px, 3)

    def draw_segments(self, segments: Optional[np.ndarray], inflation: Optional[tuple] = None, width: int = 3):
        """Draw line segment obstacles defined as (x1, y1, x2, y2)."""
        if segments is None or len(segments) == 0:
            return
        inflate_values, inflate_colors = self._parse_inflation(inflation)
        for seg in segments:
            if isinstance(seg, dict):
                if "p1" in seg and "p2" in seg:
                    p1 = np.array(seg["p1"], dtype=np.float32)
                    p2 = np.array(seg["p2"], dtype=np.float32)
                else:
                    x1 = float(seg.get("x1", 0.0))
                    y1 = float(seg.get("y1", 0.0))
                    x2 = float(seg.get("x2", 0.0))
                    y2 = float(seg.get("y2", 0.0))
                    p1 = np.array([x1, y1], dtype=np.float32)
                    p2 = np.array([x2, y2], dtype=np.float32)
            else:
                p1 = np.array([seg[0], seg[1]], dtype=np.float32)
                p2 = np.array([seg[2], seg[3]], dtype=np.float32)
            pygame.draw.line(
                self.screen,
                self.COLORS['obstacle_segment'],
                self.world_to_screen(p1),
                self.world_to_screen(p2),
                width,
            )
            if not inflate_values:
                continue
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            perp = np.array([-direction[1], direction[0]]) / norm
            for idx, value in enumerate(inflate_values):
                if value <= 0:
                    continue
                offset = perp * value
                color = inflate_colors[min(idx, len(inflate_colors) - 1)]
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(p1 + offset),
                    self.world_to_screen(p2 + offset),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(p1 - offset),
                    self.world_to_screen(p2 - offset),
                    3,
                )
    
    def draw_marker(self, position: np.ndarray, color: tuple, radius: int = 8, label: str = ""):
        """Draw marker point"""
        screen_pos = self.world_to_screen(position)
        pygame.draw.circle(self.screen, color, screen_pos, radius)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, radius, 2)
        
        if label:
            text = self.font.render(label, True, color)
            self.screen.blit(text, (screen_pos[0] + 12, screen_pos[1] - 8))
    
    def draw_robot(self, position: np.ndarray, heading: float):
        """Draw robot (triangle with heading)"""
        screen_pos = self.world_to_screen(position)
        size = 12
        
        # Calculate triangle vertices
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        
        # Front
        front = (
            screen_pos[0] + int(cos_h * size * 1.5),
            screen_pos[1] - int(sin_h * size * 1.5)
        )
        # Left rear
        left = (
            screen_pos[0] + int(-cos_h * size - sin_h * size),
            screen_pos[1] - int(-sin_h * size + cos_h * size)
        )
        # Right rear
        right = (
            screen_pos[0] + int(-cos_h * size + sin_h * size),
            screen_pos[1] - int(-sin_h * size - cos_h * size)
        )
        
        pygame.draw.polygon(self.screen, self.COLORS['robot'], [front, left, right])
        pygame.draw.polygon(self.screen, (255, 255, 255), [front, left, right], 2)
    
    def draw_human(self, position: np.ndarray):
        """Draw human (circle)"""
        screen_pos = self.world_to_screen(position)
        pygame.draw.circle(self.screen, self.COLORS['human'], screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, 10, 2)

    def draw_radius(self, position: np.ndarray, radius: float, color: tuple, width: int = 2):
        if radius is None or radius <= 0:
            return
        screen_pos = self.world_to_screen(position)
        radius_px = max(2, int(radius * self.ppm))
        pygame.draw.circle(self.screen, color, screen_pos, radius_px, width)
    
    def draw_leash(self, robot_pos: np.ndarray, human_pos: np.ndarray, tension: float):
        """Draw leash"""
        robot_screen = self.world_to_screen(robot_pos)
        human_screen = self.world_to_screen(human_pos)
        
        # Change color based on tension
        color = (
            int(200 + 55 * tension),
            int(200 - 100 * tension),
            int(200 - 100 * tension)
        )
        width = 2 if tension < 0.8 else 3
        
        pygame.draw.line(self.screen, color, robot_screen, human_screen, width)
    
    def draw_ui(self, info: dict):
        """Draw UI information"""
        y = 15
        line_height = 25
        
        # Status info
        texts = [
            f"FPS: {info.get('fps', 0):.0f}",
            f"Path Length: {info.get('path_length', 0):.1f}m",
            f"Robot Pos: ({info.get('robot_x', 0):.1f}, {info.get('robot_y', 0):.1f})",
            f"Points: {info.get('num_points', 0)}",
            f"Zoom: {self.ppm:.1f} px/m",
        ]
        mode = info.get('mode')
        if mode:
            texts.append(f"Mode: {mode}")
        
        for text in texts:
            surface = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(surface, (15, y))
            y += line_height
        
        # Recording status
        if info.get('recording', False):
            rec_text = self.font_large.render("● REC", True, self.COLORS['recording'])
            self.screen.blit(rec_text, (self.width - 100, 15))
        
        # Score display (right side)
        scores = info.get('scores', {})
        if scores and scores.get('total', 0) > 0:
            self._draw_score_panel(scores)
        
        # Control hints
        controls = info.get('controls') if info else None
        if not controls:
            controls = [
                "Arrows: Move",
                "SPACE: Record",
                "S: Save",
                "R: Reset",
                "N: New Path",
                "Scroll: Zoom",
                "ESC: Exit",
            ]
        y = self.height - len(controls) * line_height - 10
        for text in controls:
            surface = self.font.render(text, True, (150, 150, 150))
            self.screen.blit(surface, (15, y))
            y += line_height
    
    def _draw_score_panel(self, scores: dict):
        """Draw score panel on the right side"""
        panel_width = 180
        panel_height = 160
        panel_x = self.width - panel_width - 15
        panel_y = 50
        
        # Panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (35, 35, 50), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 80), panel_rect, 2, border_radius=8)
        
        # Grade color
        grade = scores.get('grade', 'F')
        grade_colors = {
            'A': (50, 255, 100),
            'B': (150, 255, 50),
            'C': (255, 255, 50),
            'D': (255, 150, 50),
            'F': (255, 80, 80),
        }
        grade_color = grade_colors.get(grade, (200, 200, 200))
        
        y = panel_y + 10
        line_height = 22
        
        # Total score with grade
        total = scores.get('total', 0)
        score_text = self.font_large.render(f"{total:.0f}", True, grade_color)
        grade_text = self.font_large.render(f" ({grade})", True, grade_color)
        self.screen.blit(score_text, (panel_x + 15, y))
        self.screen.blit(grade_text, (panel_x + 60, y))
        y += 35
        
        # Individual scores
        score_items = [
            ('Path', scores.get('path_following', 0)),
            ('Smooth', scores.get('smoothness', 0)),
            ('Complete', scores.get('completion', 0)),
            ('Leash', scores.get('leash_control', 0)),
        ]
        
        for label, value in score_items:
            # Color based on score
            if value >= 80:
                color = (100, 255, 100)
            elif value >= 60:
                color = (255, 255, 100)
            else:
                color = (255, 100, 100)
            
            label_surface = self.font.render(f"{label}:", True, (180, 180, 180))
            value_surface = self.font.render(f"{value:.0f}", True, color)
            self.screen.blit(label_surface, (panel_x + 15, y))
            self.screen.blit(value_surface, (panel_x + 110, y))
            y += line_height
    
    def render(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
        human_pos: np.ndarray,
        reference_path: Optional[np.ndarray] = None,
        robot_trajectory: Optional[np.ndarray] = None,
        human_trajectory: Optional[np.ndarray] = None,
        planned_path: Optional[np.ndarray] = None,
        lookahead_points: Optional[np.ndarray] = None,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
        obstacle_inflation: Optional[tuple] = None,
        robot_radius: Optional[float] = None,
        human_radius: Optional[float] = None,
        start_pos: Optional[np.ndarray] = None,
        end_pos: Optional[np.ndarray] = None,
        leash_tension: float = 0.0,
        info: Optional[dict] = None,
        flip: bool = True
    ):
        """Render one frame. Set flip=False to manually control display update."""
        # Update camera
        self.update_camera(robot_pos)
        
        # Clear screen
        self.screen.fill(self.COLORS['background'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw reference path
        if reference_path is not None and len(reference_path) > 0:
            self.draw_path(reference_path, self.COLORS['path_ref'], 3)

        # Draw obstacles
        if obstacles is not None and len(obstacles) > 0:
            self.draw_obstacles(obstacles, obstacle_inflation)

        # Draw segment obstacles
        if segment_obstacles is not None and len(segment_obstacles) > 0:
            self.draw_segments(segment_obstacles, obstacle_inflation)

        # Draw planned path
        if planned_path is not None and len(planned_path) > 1:
            self.draw_path(planned_path, self.COLORS['path_plan'], 2)

        # Draw lookahead points
        if lookahead_points is not None and len(lookahead_points) > 0:
            self.draw_points(lookahead_points, self.COLORS['lookahead'], 5)
        
        # Draw trajectories
        if robot_trajectory is not None and len(robot_trajectory) > 1:
            self.draw_path(np.array(robot_trajectory), self.COLORS['path_robot'], 2)
        
        if human_trajectory is not None and len(human_trajectory) > 1:
            self.draw_path(np.array(human_trajectory), self.COLORS['path_human'], 2)
        
        # Draw start and end markers
        if start_pos is not None:
            self.draw_marker(start_pos, self.COLORS['start'], 10, "Start")
        
        if end_pos is not None:
            self.draw_marker(end_pos, self.COLORS['end'], 10, "End")
        
        # Draw leash
        self.draw_leash(robot_pos, human_pos, leash_tension)

        # Draw human and robot
        self.draw_radius(robot_pos, robot_radius, self.COLORS['robot_radius'])
        self.draw_radius(human_pos, human_radius, self.COLORS['human_radius'])
        self.draw_human(human_pos)
        self.draw_robot(robot_pos, robot_heading)
        
        # Draw UI
        if info is not None:
            self.draw_ui(info)
        
        # Update display (optional)
        if flip:
            pygame.display.flip()
    
    def get_events(self) -> list:
        """Get pygame events"""
        return pygame.event.get()
    
    def tick(self, fps: int = 50) -> float:
        """Control frame rate, return actual FPS"""
        self.clock.tick(fps)
        return self.clock.get_fps()
    
    def quit(self):
        """Quit pygame"""
        pygame.quit()


if __name__ == "__main__":
    # Test visualization
    vis = Visualizer()
    
    running = True
    robot_pos = np.array([0.0, 0.0])
    human_pos = np.array([-1.0, 0.0])
    heading = 0.0
    
    while running:
        for event in vis.get_events():
            if event.type == pygame.QUIT:
                running = False
            vis.handle_zoom(event)
        
        # Simulate movement
        heading += 0.02
        robot_pos += np.array([0.05, 0.02])
        
        vis.render(
            robot_pos=robot_pos,
            robot_heading=heading,
            human_pos=human_pos,
            info={'fps': vis.tick(), 'robot_x': robot_pos[0], 'robot_y': robot_pos[1]}
        )
    
    vis.quit()
