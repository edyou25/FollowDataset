"""
Path Generator - Generate random reference paths (tactile paving style)
"""
import numpy as np
from typing import List


class PathGenerator:
    """Generate random reference paths with sharp turns like tactile paving"""
    
    def __init__(self, target_length: float = 50.0, num_segments: int = 6):
        """
        Args:
            target_length: Target path length (meters)
            num_segments: Number of straight segments
        """
        self.target_length = target_length
        self.num_segments = num_segments
    
    def generate(self) -> dict:
        """
        Generate a random reference path with sharp turns
        
        Returns:
            dict: Contains path, start, end, length
        """
        # Generate waypoints with sharp turns
        waypoints = self._generate_tactile_waypoints()
        
        # Interpolate straight line segments (keep sharp corners)
        path = self._interpolate_segments(waypoints)
        
        # Scale to target length
        path = self._scale_to_length(path)
        
        return {
            'path': path,
            'start': path[0].copy(),
            'end': path[-1].copy(),
            'length': self._compute_length(path),
            'waypoints': waypoints
        }
    
    def _generate_tactile_waypoints(self) -> np.ndarray:
        """Generate waypoints simulating tactile paving (blind path)"""
        points = [np.array([0.0, 0.0])]
        
        # Initial direction (heading angle)
        current_heading = 0.0  # Start facing +x direction
        
        for i in range(self.num_segments):
            prev = points[-1]
            
            # Segment length: varies between 5-12 meters
            segment_length = np.random.uniform(5.0, 12.0)
            
            # Move in current direction
            dx = np.cos(current_heading) * segment_length
            dy = np.sin(current_heading) * segment_length
            points.append(prev + np.array([dx, dy]))
            
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
        
        return np.array(points)
    
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
    
    def _scale_to_length(self, path: np.ndarray) -> np.ndarray:
        """Scale path to target length"""
        current_length = self._compute_length(path)
        if current_length > 0:
            scale = self.target_length / current_length
            # Scale from start point
            start = path[0].copy()
            path = (path - start) * scale + start
        return path
    
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
