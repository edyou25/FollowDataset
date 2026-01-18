"""
Path Generator - Generate random reference paths (tactile paving style)
"""
import numpy as np
from typing import Optional


class PathGenerator:
    """Generate random reference paths with sharp turns like tactile paving."""

    def __init__(
        self,
        target_length: float = 50.0,
        num_segments: int = 6,
        corridor_width: float = 2.5,
        obstacle_radius: float = 0.3,
        obstacle_jitter: float = 0.4,
        wall_miter_limit: float = 4.0,
    ):
        """
        Args:
            target_length: Target path length (meters)
            num_segments: Number of straight segments
            corridor_width: Corridor width for obstacles (meters)
            obstacle_radius: Circle obstacle radius (meters)
            obstacle_jitter: Along-path jitter for circle obstacles (meters)
            wall_miter_limit: Max miter length factor for wall joins
        """
        self.target_length = target_length
        self.num_segments = num_segments
        self.corridor_width = corridor_width
        self.obstacle_radius = obstacle_radius
        self.obstacle_jitter = obstacle_jitter
        self.wall_miter_limit = float(wall_miter_limit)
    
    def generate(self) -> dict:
        """
        Generate a random reference path with sharp turns
        
        Returns:
            dict: Contains path, start, end, length, waypoints, obstacles, segment_obstacles
        """
        # Generate waypoints with sharp turns
        waypoints = self._generate_tactile_waypoints()
        
        # Interpolate straight line segments (keep sharp corners)
        path = self._interpolate_segments(waypoints)
        
        # Scale to target length (keep waypoints in sync)
        current_length = self._compute_length(path)
        scale = self.target_length / current_length if current_length > 0 else 1.0
        start = path[0].copy()
        path = (path - start) * scale + start
        waypoints = (waypoints - start) * scale + start

        obstacles = self._generate_corridor_obstacles(waypoints)
        segment_obstacles = self._generate_segment_obstacles(waypoints)
        
        return {
            'path': path,
            'start': path[0].copy(),
            'end': path[-1].copy(),
            'length': self._compute_length(path),
            'waypoints': waypoints,
            'obstacles': obstacles,
            'segment_obstacles': segment_obstacles,
        }
    
    def _generate_tactile_waypoints(self) -> np.ndarray:
        """Generate waypoints simulating tactile paving (blind path)"""
        max_tries = 80
        min_clearance = float(self.corridor_width)
        last_points = None

        for _ in range(max_tries):
            points = [np.array([0.0, 0.0])]
            current_heading = 0.0  # Start facing +x direction
            valid = True

            for i in range(self.num_segments):
                prev = points[-1]

                # Segment length: varies between 5-12 meters
                segment_length = np.random.uniform(5.0, 12.0)

                # Move in current direction
                dx = np.cos(current_heading) * segment_length
                dy = np.sin(current_heading) * segment_length
                candidate = prev + np.array([dx, dy])

                if not self._segment_is_valid(prev, candidate, points, min_clearance):
                    valid = False
                    break

                points.append(candidate)

                # Turn for next segment (except last)
                if i < self.num_segments - 1:
                    turn_type = np.random.choice(['sharp', 'right_angle', 'slight'])

                    if turn_type == 'sharp':
                        # Sharp turn: 60-120 degrees
                        turn_angle = np.random.choice([-1, 1]) * np.random.uniform(np.pi/3, 2*np.pi/3)
                    elif turn_type == 'right_angle':
                        # Right angle turn: ~90 degrees (like real tactile paving)
                        turn_angle = np.random.choice([-1, 1]) * np.pi/2
                        turn_angle += np.random.uniform(-0.1, 0.1)  # Small variation
                    else:
                        # Slight turn: 15-45 degrees
                        turn_angle = np.random.choice([-1, 1]) * np.random.uniform(np.pi/12, np.pi/4)

                    current_heading += turn_angle

            last_points = points
            if valid:
                return np.array(points)

        print("[warn] failed to generate non-crossing waypoints; using last attempt")
        return np.array(last_points)
    
    def _interpolate_segments(self, waypoints: np.ndarray, points_per_meter: float = 10.0) -> np.ndarray:
        """Interpolate straight line segments between waypoints (preserving sharp corners)"""
        all_points = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Calculate segment length
            segment_length = np.linalg.norm(end - start)
            num_points = max(int(segment_length * points_per_meter), 2)
            
            # Linear interpolation (straight line)
            t = np.linspace(0, 1, num_points, endpoint=(i == len(waypoints) - 2))
            segment_points = start + np.outer(t, end - start)
            
            all_points.extend(segment_points.tolist())
        
        return np.array(all_points)
    
    def _compute_length(self, path: np.ndarray) -> float:
        """Compute path length"""
        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)

    def _segment_is_valid(
        self,
        start: np.ndarray,
        end: np.ndarray,
        points: list,
        min_clearance: float,
    ) -> bool:
        if len(points) < 2:
            return True
        half_width = 0.5 * float(self.corridor_width)
        clearance_sq = float(min_clearance * min_clearance)
        for i in range(len(points) - 2):
            p1 = points[i]
            p2 = points[i + 1]
            if self._segments_intersect(p1, p2, start, end):
                return False
            if self._segment_distance_sq(p1, p2, start, end) < clearance_sq:
                return False
            if self._corridor_walls_intersect(p1, p2, start, end, half_width):
                return False
        return True

    def _segments_intersect(
        self, p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
    ) -> bool:
        def orientation(a, b, c) -> int:
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if abs(val) < 1e-9:
                return 0
            return 1 if val > 0 else 2

        def on_segment(a, b, c) -> bool:
            return (
                min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
                and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
            )

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
        return False

    def _segment_distance_sq(
        self, p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
    ) -> float:
        if self._segments_intersect(p1, p2, q1, q2):
            return 0.0

        return min(
            self._point_segment_dist_sq(p1, q1, q2),
            self._point_segment_dist_sq(p2, q1, q2),
            self._point_segment_dist_sq(q1, p1, p2),
            self._point_segment_dist_sq(q2, p1, p2),
        )

    def _point_segment_dist_sq(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        ab = p2 - p1
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            diff = point - p1
            return float(np.dot(diff, diff))
        t = float(np.dot(point - p1, ab)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        closest = p1 + t * ab
        diff = point - closest
        return float(np.dot(diff, diff))

    def _corridor_walls_intersect(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
        half_width: float,
    ) -> bool:
        offsets_a = self._offset_segment(p1, p2, half_width)
        offsets_b = self._offset_segment(p1, p2, -half_width)
        offsets_c = self._offset_segment(q1, q2, half_width)
        offsets_d = self._offset_segment(q1, q2, -half_width)
        candidates_a = [seg for seg in (offsets_a, offsets_b) if seg is not None]
        candidates_b = [seg for seg in (offsets_c, offsets_d) if seg is not None]
        for seg_a in candidates_a:
            for seg_b in candidates_b:
                if self._segments_intersect(seg_a[0], seg_a[1], seg_b[0], seg_b[1]):
                    return True
        return False

    def _offset_segment(
        self, p1: np.ndarray, p2: np.ndarray, offset: float
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        perp = np.array([-direction[1], direction[0]]) / norm
        shift = perp * float(offset)
        return p1 + shift, p2 + shift
    
    def _generate_corridor_obstacles(self, waypoints: np.ndarray) -> np.ndarray:
        """Generate simple circular obstacles around waypoints to form a corridor."""
        if waypoints is None or len(waypoints) < 2:
            return np.zeros((0, 3), dtype=np.float32)

        half_width = 0.5 * float(self.corridor_width)
        obstacles = []
        for i, point in enumerate(waypoints):
            if i == 0:
                direction = waypoints[i + 1] - waypoints[i]
            elif i == len(waypoints) - 1:
                direction = waypoints[i] - waypoints[i - 1]
            else:
                direction = waypoints[i + 1] - waypoints[i - 1]

            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            perp = np.array([-direction[1], direction[0]])
            jitter = direction * np.random.uniform(-self.obstacle_jitter, self.obstacle_jitter)

            for side in (-1.0, 1.0):
                pos = point + perp * half_width * side + jitter
                obstacles.append([float(pos[0]), float(pos[1]), float(self.obstacle_radius)])

        return np.asarray(obstacles, dtype=np.float32)

    def _generate_segment_obstacles(self, waypoints: np.ndarray) -> np.ndarray:
        """Generate corridor walls as line segments."""
        if waypoints is None or len(waypoints) < 2:
            return np.zeros((0, 4), dtype=np.float32)

        half_width = 0.5 * float(self.corridor_width)
        segments = []
        directions = []
        for i in range(len(waypoints) - 1):
            direction = waypoints[i + 1] - waypoints[i]
            norm = np.linalg.norm(direction)
            directions.append(direction / norm if norm >= 1e-6 else None)

        left_points = self._build_wall_polyline(waypoints, directions, half_width, side=1.0)
        right_points = self._build_wall_polyline(waypoints, directions, half_width, side=-1.0)

        for pts in (left_points, right_points):
            for i in range(len(pts) - 1):
                if np.linalg.norm(pts[i + 1] - pts[i]) < 1e-6:
                    continue
                segments.append([pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]])

        return np.asarray(segments, dtype=np.float32)

    def _build_wall_polyline(
        self,
        waypoints: np.ndarray,
        directions: list,
        half_width: float,
        side: float,
    ) -> list:
        points = []
        n_seg = len(waypoints) - 1
        if n_seg <= 0:
            return points

        start_dir = directions[0]
        if start_dir is None:
            return points
        start_offset = np.array([-start_dir[1], start_dir[0]]) * side * half_width
        points.append(waypoints[0] + start_offset)

        for i in range(1, n_seg):
            prev_dir = directions[i - 1]
            curr_dir = directions[i]
            if prev_dir is None or curr_dir is None:
                continue
            p_prev = waypoints[i] + np.array([-prev_dir[1], prev_dir[0]]) * side * half_width
            p_curr = waypoints[i] + np.array([-curr_dir[1], curr_dir[0]]) * side * half_width
            inter = self._line_intersection(p_prev, prev_dir, p_curr, curr_dir)
            if inter is None:
                points.append(p_prev)
                points.append(p_curr)
                continue
            if np.linalg.norm(inter - waypoints[i]) > self.wall_miter_limit * half_width:
                points.append(p_prev)
                points.append(p_curr)
            else:
                points.append(inter)

        end_dir = directions[-1]
        if end_dir is not None:
            end_offset = np.array([-end_dir[1], end_dir[0]]) * side * half_width
            points.append(waypoints[-1] + end_offset)
        return points

    def _line_intersection(
        self,
        p1: np.ndarray,
        d1: np.ndarray,
        p2: np.ndarray,
        d2: np.ndarray,
    ) -> Optional[np.ndarray]:
        def cross(a, b) -> float:
            return float(a[0] * b[1] - a[1] * b[0])

        denom = cross(d1, d2)
        if abs(denom) < 1e-9:
            return None
        t = cross(p2 - p1, d2) / denom
        return p1 + t * d1
    
    def get_closest_point_on_path(self, path: np.ndarray, position: np.ndarray) -> tuple:
        """
        Get closest point on path to given position
        
        Returns:
            tuple: (closest_point, index, distance)
        """
        diffs = path - position
        distances = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(distances)
        return path[idx], idx, distances[idx]


if __name__ == "__main__":
    # Test path generation
    generator = PathGenerator(target_length=50.0)
    
    for i in range(3):
        result = generator.generate()
        print(f"\nPath {i+1}:")
        print(f"  Length: {result['length']:.2f}m")
        print(f"  Segments: {len(result['waypoints']) - 1}")
        print(f"  Start: {result['start']}")
        print(f"  End: {result['end']}")
