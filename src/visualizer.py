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
        'obstacle_obs': (255, 60, 60),
        'clear_safe': (80, 220, 120),
        'clear_warn': (255, 200, 80),
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

        # Layer visibility (click legend to toggle)
        self.layer_visibility = {
            "grid": True,
            "reference_path": True,
            "planned_path": True,
            "lookahead_points": True,
            "robot_trajectory": True,
            "human_trajectory": True,
            "obstacles": True,
            "segment_obstacles": True,
            "obs_obstacles": True,
            "obs_segment_vectors": True,
            "human_clearance": True,
            "start_end": True,
            "leash": True,
            "agent_radii": True,
        }
        self._legend_hitboxes: dict[str, pygame.Rect] = {}
        self._legend_hover_key: Optional[str] = None
        
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

    def handle_event(self, event: pygame.event.Event):
        """Handle mouse events (zoom + legend toggles)."""
        self.handle_zoom(event)
        self._handle_legend_event(event)

    def _legend_items(self) -> list[dict]:
        return [
            {"key": "grid", "label": "Grid", "color": self.COLORS["grid"]},
            {"key": "reference_path", "label": "Ref Path", "color": self.COLORS["path_ref"]},
            {"key": "planned_path", "label": "Planned Path", "color": self.COLORS["path_plan"]},
            {"key": "lookahead_points", "label": "Lookahead", "color": self.COLORS["lookahead"]},
            {"key": "robot_trajectory", "label": "Robot Trail", "color": self.COLORS["path_robot"]},
            {"key": "human_trajectory", "label": "Human Trail", "color": self.COLORS["path_human"]},
            {"key": "obstacles", "label": "Circle Obstacles", "color": self.COLORS["obstacle"]},
            {"key": "segment_obstacles", "label": "Wall Segments", "color": self.COLORS["obstacle_segment"]},
            {"key": "obs_obstacles", "label": "Obs Highlight", "color": self.COLORS["obstacle_obs"]},
            {"key": "obs_segment_vectors", "label": "Obs Vectors", "color": self.COLORS["obstacle_obs"]},
            {"key": "human_clearance", "label": "Human Clearance", "color": self.COLORS["clear_safe"]},
            {"key": "start_end", "label": "Start/End", "color": self.COLORS["start"]},
            {"key": "leash", "label": "Leash", "color": self.COLORS["leash"]},
            {"key": "agent_radii", "label": "Agent Radii", "color": self.COLORS["robot_radius"]},
        ]

    def _handle_legend_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            hover = None
            for key, rect in self._legend_hitboxes.items():
                if rect.collidepoint(pos):
                    hover = key
                    break
            self._legend_hover_key = hover
            return

        if event.type != pygame.MOUSEBUTTONDOWN:
            return
        if getattr(event, "button", None) != 1:
            return
        pos = getattr(event, "pos", None)
        if pos is None:
            return
        for key, rect in self._legend_hitboxes.items():
            if rect.collidepoint(pos):
                self.layer_visibility[key] = not bool(self.layer_visibility.get(key, True))
                return

    def draw_legend(self, x: Optional[int] = None, y: Optional[int] = None):
        items = self._legend_items()
        if not items:
            return

        panel_width = 260
        header_h = 26
        row_h = 24
        pad = 10
        panel_height = header_h + len(items) * row_h + pad

        if x is None:
            x = self.width - panel_width - 15
        if y is None:
            y = 230

        panel_rect = pygame.Rect(x, y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (35, 35, 50), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (60, 60, 80), panel_rect, 2, border_radius=8)

        title = self.font.render("Legend (click to toggle)", True, (180, 180, 180))
        self.screen.blit(title, (x + 10, y + 6))

        self._legend_hitboxes = {}
        for i, item in enumerate(items):
            key = item["key"]
            label = item["label"]
            color = item["color"]
            enabled = bool(self.layer_visibility.get(key, True))

            row_y = y + header_h + i * row_h
            row_rect = pygame.Rect(x + 8, row_y, panel_width - 16, row_h)
            self._legend_hitboxes[key] = row_rect

            if self._legend_hover_key == key:
                pygame.draw.rect(self.screen, (50, 50, 70), row_rect, border_radius=6)

            box = pygame.Rect(x + 12, row_y + 5, 14, 14)
            pygame.draw.rect(self.screen, (220, 220, 220), box, 1)
            fill_color = color if enabled else (90, 90, 110)
            pygame.draw.rect(self.screen, fill_color, box.inflate(-2, -2))

            if not enabled:
                pygame.draw.line(
                    self.screen, (220, 220, 220), box.topleft, box.bottomright, 2
                )
                pygame.draw.line(
                    self.screen, (220, 220, 220), box.topright, box.bottomleft, 2
                )

            label_surface = self.font.render(label, True, (220, 220, 220))
            self.screen.blit(label_surface, (x + 34, row_y + 2))
    
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

    def draw_observation_obstacles(
        self,
        obstacles: Optional[np.ndarray] = None,
        segment_obstacles: Optional[np.ndarray] = None,
        circle_width: int = 4,
        segment_width: int = 5,
    ):
        """Highlight obstacles that are currently used as model observations."""
        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                if isinstance(obs, dict):
                    x = float(obs.get("x", 0.0))
                    y = float(obs.get("y", 0.0))
                    r = float(obs.get("r", 0.0))
                else:
                    x = float(obs[0])
                    y = float(obs[1])
                    r = float(obs[2])
                screen_pos = self.world_to_screen(np.array([x, y], dtype=np.float32))
                radius_px = max(2, int(r * self.ppm))
                pygame.draw.circle(
                    self.screen,
                    self.COLORS["obstacle_obs"],
                    screen_pos,
                    radius_px,
                    circle_width,
                )

        if segment_obstacles is not None and len(segment_obstacles) > 0:
            for seg in segment_obstacles:
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
                    self.COLORS["obstacle_obs"],
                    self.world_to_screen(p1),
                    self.world_to_screen(p2),
                    segment_width,
                )

    def draw_observation_segment_vectors(
        self,
        robot_pos: np.ndarray,
        closest_points: Optional[np.ndarray] = None,
        directions: Optional[np.ndarray] = None,
        arrow_length: float = 1.0,
    ):
        """Draw closest-point vector and wall direction for observed segments."""
        if closest_points is None or len(closest_points) == 0:
            return

        pts = np.asarray(closest_points, dtype=np.float32)
        dirs = None
        if directions is not None and len(directions) == len(closest_points):
            dirs = np.asarray(directions, dtype=np.float32)

        robot_screen = self.world_to_screen(robot_pos)
        color = self.COLORS["obstacle_obs"]
        for i, cp in enumerate(pts):
            cp_screen = self.world_to_screen(cp)
            pygame.draw.line(self.screen, color, robot_screen, cp_screen, 2)
            pygame.draw.circle(self.screen, color, cp_screen, 6, 2)

            if dirs is None:
                continue
            d = dirs[i]
            norm = float(np.linalg.norm(d))
            if norm < 1e-6:
                continue
            d = d / norm
            end = cp + d * float(arrow_length)
            end_screen = self.world_to_screen(end)
            pygame.draw.line(self.screen, color, cp_screen, end_screen, 3)

            # arrow head
            head_len = 0.25
            head_w = 0.12
            perp = np.array([-d[1], d[0]], dtype=np.float32)
            left = end - d * head_len + perp * head_w
            right = end - d * head_len - perp * head_w
            pygame.draw.line(self.screen, color, end_screen, self.world_to_screen(left), 3)
            pygame.draw.line(self.screen, color, end_screen, self.world_to_screen(right), 3)

    def draw_human_clearance(
        self,
        human_pos: np.ndarray,
        human_radius: Optional[float],
        obs_obstacles: Optional[np.ndarray] = None,
        obs_segment_obstacles: Optional[np.ndarray] = None,
        warn_clearance: float = 0.2,
    ):
        """Visualize the minimum-distance (clearance) from the human to observed obstacles."""
        if human_radius is None or human_radius <= 0:
            return

        human_pos = np.asarray(human_pos, dtype=np.float32)
        hr = float(human_radius)
        human_screen = self.world_to_screen(human_pos)

        def clearance_color(clearance: float) -> tuple[int, int, int]:
            if clearance < 0.0:
                return self.COLORS["obstacle_obs"]
            if clearance < warn_clearance:
                return self.COLORS["clear_warn"]
            return self.COLORS["clear_safe"]

        def clearance_width(clearance: float) -> int:
            if clearance < 0.0:
                return 4
            if clearance < warn_clearance:
                return 3
            return 2

        if obs_obstacles is not None and len(obs_obstacles) > 0:
            for obs in obs_obstacles:
                if isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    r = float(obs.get("r", 0.0))
                else:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    r = float(obs[2])
                center = np.array([ox, oy], dtype=np.float32)
                d = human_pos - center
                dist = float(np.linalg.norm(d))
                if dist < 1e-6:
                    continue
                dir_obs_to_h = d / dist
                p_obs = center + dir_obs_to_h * float(r)
                p_h = human_pos - dir_obs_to_h * hr
                clearance = dist - float(r + hr)
                color = clearance_color(clearance)
                width = clearance_width(clearance)
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(p_obs),
                    self.world_to_screen(p_h),
                    width,
                )
                pygame.draw.circle(self.screen, color, self.world_to_screen(p_obs), 4, 1)
                pygame.draw.circle(self.screen, color, self.world_to_screen(p_h), 4, 1)

        if obs_segment_obstacles is not None and len(obs_segment_obstacles) > 0:
            for seg in obs_segment_obstacles:
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

                ab = p2 - p1
                denom = float(np.dot(ab, ab))
                if denom < 1e-12:
                    closest = p1
                else:
                    t = float(np.dot(human_pos - p1, ab)) / denom
                    t = float(np.clip(t, 0.0, 1.0))
                    closest = p1 + t * ab

                d = human_pos - closest
                dist = float(np.linalg.norm(d))
                if dist < 1e-6:
                    continue
                dir_wall_to_h = d / dist
                p_h = human_pos - dir_wall_to_h * hr
                clearance = dist - hr
                color = clearance_color(clearance)
                width = clearance_width(clearance)
                pygame.draw.line(
                    self.screen,
                    color,
                    self.world_to_screen(closest),
                    self.world_to_screen(p_h),
                    width,
                )
                pygame.draw.circle(self.screen, color, self.world_to_screen(closest), 4, 1)
                pygame.draw.circle(self.screen, color, self.world_to_screen(p_h), 4, 1)

        # A small center dot to make it easier to see which clearance vectors belong to human
        pygame.draw.circle(self.screen, self.COLORS["human"], human_screen, 3)
    
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
        obs_obstacles: Optional[np.ndarray] = None,
        obs_segment_obstacles: Optional[np.ndarray] = None,
        obs_segment_closest_points: Optional[np.ndarray] = None,
        obs_segment_dirs: Optional[np.ndarray] = None,
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
        if self.layer_visibility.get("grid", True):
            self.draw_grid()
        
        # Draw reference path
        if reference_path is not None and len(reference_path) > 0:
            if self.layer_visibility.get("reference_path", True):
                self.draw_path(reference_path, self.COLORS['path_ref'], 3)

        # Draw obstacles
        if obstacles is not None and len(obstacles) > 0:
            if self.layer_visibility.get("obstacles", True):
                self.draw_obstacles(obstacles, obstacle_inflation)

        # Draw segment obstacles
        if segment_obstacles is not None and len(segment_obstacles) > 0:
            if self.layer_visibility.get("segment_obstacles", True):
                self.draw_segments(segment_obstacles, obstacle_inflation)

        # Highlight obstacles used as observation
        if (
            (obs_obstacles is not None and len(obs_obstacles) > 0)
            or (obs_segment_obstacles is not None and len(obs_segment_obstacles) > 0)
        ):
            if self.layer_visibility.get("obs_obstacles", True):
                self.draw_observation_obstacles(obs_obstacles, obs_segment_obstacles)
        if obs_segment_closest_points is not None and len(obs_segment_closest_points) > 0:
            if self.layer_visibility.get("obs_segment_vectors", True):
                self.draw_observation_segment_vectors(
                    robot_pos, obs_segment_closest_points, obs_segment_dirs
                )
        if (
            human_radius is not None
            and (
                (obs_obstacles is not None and len(obs_obstacles) > 0)
                or (obs_segment_obstacles is not None and len(obs_segment_obstacles) > 0)
            )
        ):
            if self.layer_visibility.get("human_clearance", True):
                self.draw_human_clearance(
                    human_pos=human_pos,
                    human_radius=human_radius,
                    obs_obstacles=obs_obstacles,
                    obs_segment_obstacles=obs_segment_obstacles,
                )

        # Draw planned path
        if planned_path is not None and len(planned_path) > 1:
            if self.layer_visibility.get("planned_path", True):
                self.draw_path(planned_path, self.COLORS['path_plan'], 2)

        # Draw lookahead points
        if lookahead_points is not None and len(lookahead_points) > 0:
            if self.layer_visibility.get("lookahead_points", True):
                self.draw_points(lookahead_points, self.COLORS['lookahead'], 5)
        
        # Draw trajectories
        if robot_trajectory is not None and len(robot_trajectory) > 1:
            if self.layer_visibility.get("robot_trajectory", True):
                self.draw_path(np.array(robot_trajectory), self.COLORS['path_robot'], 2)
        
        if human_trajectory is not None and len(human_trajectory) > 1:
            if self.layer_visibility.get("human_trajectory", True):
                self.draw_path(np.array(human_trajectory), self.COLORS['path_human'], 2)
        
        # Draw start and end markers
        if start_pos is not None:
            if self.layer_visibility.get("start_end", True):
                self.draw_marker(start_pos, self.COLORS['start'], 10, "Start")
        
        if end_pos is not None:
            if self.layer_visibility.get("start_end", True):
                self.draw_marker(end_pos, self.COLORS['end'], 10, "End")
        
        # Draw leash
        if self.layer_visibility.get("leash", True):
            self.draw_leash(robot_pos, human_pos, leash_tension)

        # Draw human and robot
        if self.layer_visibility.get("agent_radii", True):
            self.draw_radius(robot_pos, robot_radius, self.COLORS['robot_radius'])
            self.draw_radius(human_pos, human_radius, self.COLORS['human_radius'])
        self.draw_human(human_pos)
        self.draw_robot(robot_pos, robot_heading)
        
        # Draw UI
        if info is not None:
            self.draw_ui(info)
        self.draw_legend()
        
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
            vis.handle_event(event)
        
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
