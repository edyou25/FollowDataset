"""
Scoring System - Evaluate trajectory quality for data collection
"""
import numpy as np
from typing import Optional, List, Dict


class TrajectoryScorer:
    """Evaluate trajectory quality for imitation learning data"""
    
    def __init__(self, reference_path: np.ndarray, leash_length: float = 1.5):
        """
        Args:
            reference_path: Reference path to follow
            leash_length: Leash length for tension calculation
        """
        self.reference_path = reference_path
        self.leash_length = leash_length
        
        # Accumulated data for scoring
        self.robot_positions: List[np.ndarray] = []
        self.human_positions: List[np.ndarray] = []
        self.path_deviations: List[float] = []
        self.leash_tensions: List[float] = []
        
        # Reference path info
        self.path_length = self._compute_path_length(reference_path)
        self.end_pos = reference_path[-1]
    
    def reset(self, reference_path: Optional[np.ndarray] = None):
        """Reset scorer for new episode"""
        if reference_path is not None:
            self.reference_path = reference_path
            self.path_length = self._compute_path_length(reference_path)
            self.end_pos = reference_path[-1]
        
        self.robot_positions = []
        self.human_positions = []
        self.path_deviations = []
        self.leash_tensions = []
    
    def update(self, robot_pos: np.ndarray, human_pos: np.ndarray):
        """Update with new frame data"""
        self.robot_positions.append(robot_pos.copy())
        self.human_positions.append(human_pos.copy())
        
        # Calculate path deviation
        deviation = self._get_min_distance_to_path(human_pos)
        self.path_deviations.append(deviation)
        
        # Calculate leash tension
        leash_dist = np.linalg.norm(robot_pos - human_pos)
        tension = leash_dist / self.leash_length
        self.leash_tensions.append(tension)
    
    def get_scores(self) -> Dict[str, float]:
        """
        Calculate all scores
        
        Returns:
            Dict with individual scores and total score (0-100)
        """
        if len(self.robot_positions) < 10:
            return {
                'path_following': 0.0,
                'smoothness': 0.0,
                'completion': 0.0,
                'leash_control': 0.0,
                'total': 0.0,
                'grade': 'N/A'
            }
        
        # 1. Path Following Score (0-100)
        # How well robot follows reference path
        avg_deviation = np.mean(self.path_deviations)
        # Good: <0.5m deviation, Bad: >3m deviation
        path_score = max(0, 100 - avg_deviation * 33.3)
        
        # 2. Smoothness Score (0-100)
        # Trajectory smoothness (low jerk)
        robot_arr = np.array(self.robot_positions)
        smoothness_score = self._calculate_smoothness(robot_arr)
        
        # 3. Completion Score (0-100)
        # How close to the end point
        final_pos = self.robot_positions[-1]
        dist_to_end = np.linalg.norm(final_pos - self.end_pos)
        # Good: <1m from end, Bad: >10m from end
        completion_score = max(0, 100 - dist_to_end * 10)
        
        # 4. Leash Control Score (0-100)
        # Human following properly (tension not too high/low)
        avg_tension = np.mean(self.leash_tensions)
        # Ideal tension: 0.5-0.8
        if 0.4 <= avg_tension <= 0.85:
            leash_score = 100.0
        elif avg_tension < 0.4:
            leash_score = max(0, avg_tension / 0.4 * 100)
        else:
            leash_score = max(0, 100 - (avg_tension - 0.85) * 200)
        
        # Total Score (weighted average)
        weights = {
            'path_following': 0.35,
            'smoothness': 0.20,
            'completion': 0.30,
            'leash_control': 0.15
        }
        
        total = (
            path_score * weights['path_following'] +
            smoothness_score * weights['smoothness'] +
            completion_score * weights['completion'] +
            leash_score * weights['leash_control']
        )
        
        # Grade
        if total >= 90:
            grade = 'A'
        elif total >= 80:
            grade = 'B'
        elif total >= 70:
            grade = 'C'
        elif total >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'path_following': round(path_score, 1),
            'smoothness': round(smoothness_score, 1),
            'completion': round(completion_score, 1),
            'leash_control': round(leash_score, 1),
            'total': round(total, 1),
            'grade': grade
        }
    
    def _get_min_distance_to_path(self, position: np.ndarray) -> float:
        """Get minimum distance from position to reference path"""
        diffs = self.reference_path - position
        distances = np.linalg.norm(diffs, axis=1)
        return np.min(distances)
    
    def _calculate_smoothness(self, trajectory: np.ndarray) -> float:
        """Calculate smoothness score based on acceleration changes"""
        if len(trajectory) < 4:
            return 50.0
        
        # Calculate velocity
        velocity = np.diff(trajectory, axis=0)
        
        # Calculate acceleration
        acceleration = np.diff(velocity, axis=0)
        
        # Calculate jerk (change in acceleration)
        jerk = np.diff(acceleration, axis=0)
        
        # RMS jerk
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        rms_jerk = np.sqrt(np.mean(jerk_magnitude ** 2))
        
        # Convert to score (lower jerk = higher score)
        # Good: <0.01, Bad: >0.1
        score = max(0, 100 - rms_jerk * 1000)
        return min(100, score)
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute path length"""
        diffs = np.diff(path, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def get_summary(self) -> str:
        """Get human-readable score summary"""
        scores = self.get_scores()
        return (
            f"Score: {scores['total']:.0f}/100 ({scores['grade']})\n"
            f"  Path: {scores['path_following']:.0f}  "
            f"Smooth: {scores['smoothness']:.0f}  "
            f"Complete: {scores['completion']:.0f}  "
            f"Leash: {scores['leash_control']:.0f}"
        )


if __name__ == "__main__":
    # Test scoring
    import numpy as np
    
    # Create simple reference path
    ref_path = np.array([[i, 0] for i in range(50)])
    
    scorer = TrajectoryScorer(ref_path)
    
    # Simulate good trajectory
    for i in range(500):
        robot_pos = np.array([i * 0.1, np.sin(i * 0.01) * 0.2])
        human_pos = robot_pos - np.array([1.2, 0])
        scorer.update(robot_pos, human_pos)
    
    print("Good trajectory:")
    print(scorer.get_summary())
    
    # Reset and simulate bad trajectory
    scorer.reset()
    for i in range(500):
        robot_pos = np.array([i * 0.05, np.sin(i * 0.1) * 3])  # Large deviation
        human_pos = robot_pos - np.array([1.5, 0])  # Tight leash
        scorer.update(robot_pos, human_pos)
    
    print("\nBad trajectory:")
    print(scorer.get_summary())

