"""
物理模型 - 机器人运动 + 绳子牵引人类跟随
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    heading: float = 0.0  # 朝向角度（弧度）
    
    def copy(self):
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=self.heading
        )




class PhysicsEngine:
    """物理引擎 - 处理机器人控制和人类跟随"""
    
    def __init__(
        self,
        leash_length: float = 1.5,  # 绳子长度（米）
        robot_speed: float = 1.5,   # 机器人移动速度（米/秒）
        turn_speed: float = 1.5,    # 机器人转向速度（弧度/秒，降低以匹配较慢的运动）
        human_drag: float = 0.9,    # 人类阻尼系数
        dt: float = 0.02,           # 时间步长（秒）
        robot_radius: float = 0.1,  # 机器人半径（米）
        human_radius: float = 0.1   # 人半径（米）
    ):
        self.leash_length = leash_length
        self.robot_speed = robot_speed
        self.turn_speed = turn_speed
        self.human_drag = human_drag
        self.dt = dt
        self.robot_radius = robot_radius
        self.human_radius = human_radius
        
        # 状态
        self.robot = State()
        self.human = State()
        
        # 控制输入
        self.forward_input = 0.0  # -1 到 1
        self.turn_input = 0.0     # -1 到 1
        self.bre = False
        self.random_angle = 0.0
    
    def reset(self, start_position: Optional[np.ndarray] = None):
        """重置物理状态"""
        if start_position is None:
            start_position = np.array([0.0, 0.0])
        
        self.robot = State(
            position=start_position.copy(),
            velocity=np.array([0.0, 0.0]),
            heading=0.0
        )
        
        # 人在机器人后方
        human_offset = np.array([-self.leash_length * 0.8, 0.0])
        self.human = State(
            position=start_position + human_offset,
            velocity=np.array([0.0, 0.0]),
            heading=0.0
        )
        
        self.forward_input = 0.0
        self.turn_input = 0.0
        self.bre = False
    
    def set_control(self, forward: float, turn: float, bre: bool = False):
        """设置控制输入"""
        self.forward_input = np.clip(forward, -1.0, 1.0)
        self.turn_input = np.clip(turn, -1.0, 1.0)
        if bre != self.bre:
            magnitude = np.random.uniform(15.0, 25.0)
            self.random_angle = np.deg2rad(np.random.choice([-magnitude, magnitude]))
        self.bre = bre
    
    def step(self) -> tuple:
        """
        执行一步物理模拟
        
        Returns:
            tuple: (robot_State, human_State)
        """
        
        return self.model5()
    
    def model1(self) -> tuple:
        """
        drag model
        
        Returns:
            tuple: (robot_State, human_State)
        """
        # 1. 更新机器人朝向
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        
        # 2. 计算机器人速度（基于朝向）
        direction = np.array([
            np.cos(self.robot.heading),
            np.sin(self.robot.heading)
        ])
        target_velocity = direction * self.forward_input * self.robot_speed
        
        # 平滑速度变化
        self.robot.velocity = self.robot.velocity * 0.8 + target_velocity * 0.2
        
        # 3. 更新机器人位置
        self.robot.position = self.robot.position + self.robot.velocity * self.dt
        
        # 4. 更新人类位置（绳子牵引）
        self._human_drag()
        
        return self.robot.copy(), self.human.copy()
    
    def model2(self) -> tuple:
        """
        drag model
        
        Returns:
            tuple: (robot_State, human_State)
        """
        # 1. 更新机器人朝向
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        
        # 2. 计算机器人速度（基于朝向）
        direction = np.array([
            np.cos(self.robot.heading),
            np.sin(self.robot.heading)
        ])
        target_velocity = direction * self.forward_input * self.robot_speed
        
        # 平滑速度变化
        self.robot.velocity = self.robot.velocity * 0.8 + target_velocity * 0.2
        
        # 3. 更新机器人位置
        self.robot.position = self.robot.position + self.robot.velocity * self.dt
        
        # 4. 更新人类位置（绳子牵引）
        self._human_drag()
        
        return self.robot.copy(), self.human.copy()
    def model3(self) -> tuple:
        """
        drag model
        
        Returns:
            tuple: (robot_State, human_State)
        """
        # 1. 更新机器人朝向
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        
        # 2. 计算机器人速度（基于朝向）
        direction = np.array([
            np.cos(self.robot.heading),
            np.sin(self.robot.heading)
        ])
        target_velocity = direction * self.forward_input * self.robot_speed
        
        # 平滑速度变化
        self.robot.velocity = self.robot.velocity * 0.8 + target_velocity * 0.2
        
        # 3. 更新机器人位置
        self.robot.position = self.robot.position + self.robot.velocity * self.dt
        
        # 4. 更新人类位置（绳子牵引）
        self._human_rigid_offset()
        
        return self.robot.copy(), self.human.copy()
    def model4(self) -> tuple:
        """
        drag model
        
        Returns:
            tuple: (robot_State, human_State)
        """
        # 1. 更新机器人朝向
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        
        # 2. 计算机器人速度（基于朝向）
        direction = np.array([
            np.cos(self.robot.heading),
            np.sin(self.robot.heading)
        ])
        target_velocity = direction * self.forward_input * self.robot_speed
        
        # 平滑速度变化
        self.robot.velocity = self.robot.velocity * 0.8 + target_velocity * 0.2
        
        # 3. 更新机器人位置
        self.robot.position = self.robot.position + self.robot.velocity * self.dt
        
        # 4. 更新人类位置（绳子牵引）
        self._human_delayed_harness()
        
        return self.robot.copy(), self.human.copy()
    
    def model5(self) -> tuple:
        """
        drag model
        
        Returns:
            tuple: (robot_State, human_State)
        """
        # 1. 更新机器人朝向
        self.robot.heading += self.turn_input * self.turn_speed * self.dt
        
        # 2. 计算机器人速度（基于朝向）
        direction = np.array([
            np.cos(self.robot.heading),
            np.sin(self.robot.heading)
        ])
        target_velocity = direction * self.forward_input * self.robot_speed
        
        # 平滑速度变化
        self.robot.velocity = self.robot.velocity * 0.8 + target_velocity * 0.2
        
        # 3. 更新机器人位置
        self.robot.position = self.robot.position + self.robot.velocity * self.dt
        
        # 4. 更新人类位置（绳子牵引）
        self._human_drag()
        robot_to_human = self.human.position - self.robot.position
        distance = np.linalg.norm(robot_to_human)
        if distance > self.leash_length and self.bre:
            # correction = robot_to_human / distance * (distance - self.leash_length) * 0.5
            # self.robot.position += correction
            # self.human.position -= correction

            leash_dir = robot_to_human / distance
            human_position1 = self.robot.position + leash_dir * self.leash_length
            robot_position1 = self.robot.position

            human_position2 = self.human.position
            robot_position2 = self.human.position - leash_dir * self.leash_length

            k = .5
            self.robot.position = (k*robot_position1 + (1-k)*robot_position2) 
            self.human.position = (k*human_position1 + (1-k)*human_position2) 

            relative_velocity = self.human.velocity - self.robot.velocity
            radial_velocity = float(np.dot(relative_velocity, leash_dir))
            tangential_velocity = relative_velocity - radial_velocity * leash_dir
            unwanted_velocity = 0.8 * tangential_velocity + max(0.0, radial_velocity) * leash_dir
            self.robot.velocity += 0.5 * unwanted_velocity
            self.human.velocity -= 0.5 * unwanted_velocity
        
        return self.robot.copy(), self.human.copy()    
    
    def _human_drag(self):
        """更新人类位置 - 绳子牵引物理"""
        # 计算机器人到人的向量
        robot_to_human = self.human.position - self.robot.position
        distance = np.linalg.norm(robot_to_human)
        R = np.array([
                    [np.cos(self.random_angle), -np.sin(self.random_angle)],
                    [np.sin(self.random_angle),  np.cos(self.random_angle)],
                ])
        if distance > 1e-6:
            # 绳子方向（从机器人指向人）
            leash_dir = robot_to_human / distance
            
            if distance > self.leash_length:
                # 绳子绷紧 - 人被拉向机器人
                # 计算需要移动的距离
                pull_distance = distance - self.leash_length
                
                # 人被拉向机器人方向
                pull_force = -leash_dir * pull_distance * 10.0  # 弹性系数
                
                # 更新人的速度
                self.human.velocity = self.human.velocity + pull_force * self.dt
                if self.bre:
                    self.human.velocity = R @ self.human.velocity
                
            # 应用阻尼
            self.human.velocity = self.human.velocity * self.human_drag
            # if self.bre:
            #     self.human.velocity = self.human.velocity * 0.0
            
            # 更新人的位置
            self.human.position = self.human.position + self.human.velocity * self.dt
            
            # 硬约束：确保不超过绳子长度
            robot_to_human = self.human.position - self.robot.position
            distance = np.linalg.norm(robot_to_human)
            if distance > self.leash_length and not self.bre:
            # if distance > self.leash_length:
                leash_dir = robot_to_human / distance
                self.human.position = self.robot.position + leash_dir * self.leash_length
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


    def _rot2d(self, theta: float) -> np.ndarray:
        """2D rotation matrix."""
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, -s],
            [s,  c],
        ], dtype=float)


    def _human_rigid_offset(self):
        """Update human pose using fixed rigid robot-human offset."""
        offset_xy = np.array([-.789359, -.481186], dtype=float)
        offset_theta = 0.269089

        robot_theta = self.robot.heading
        robot_R = self._rot2d(robot_theta)

        old_position = self.human.position.copy()

        self.human.position = self.robot.position + robot_R @ offset_xy
        self.human.velocity = (self.human.position - old_position) / max(self.dt, 1e-6)

        self.human.heading = self._wrap_angle(robot_theta + offset_theta)


    def _human_delayed_harness(self):
        """Update human pose using delayed harness model."""
        offset_xy = np.array([-0.789359, -0.481186], dtype=float)
        offset_theta = 0.269089
        alpha = 0.754387

        robot_theta = self.robot.heading
        robot_R = self._rot2d(robot_theta)

        old_position = self.human.position.copy()
        old_heading = self.human.heading

        rigid_target_position = self.robot.position + robot_R @ offset_xy
        rigid_target_heading = self._wrap_angle(robot_theta + offset_theta)

        self.human.position = (
            alpha * old_position
            + (1.0 - alpha) * rigid_target_position
        )
        self.human.velocity = (self.human.position - old_position) / max(self.dt, 1e-6)

        heading_error = self._wrap_angle(rigid_target_heading - old_heading)
        self.human.heading = self._wrap_angle(
            old_heading + (1.0 - alpha) * heading_error
        )    
    
    def get_leash_tension(self) -> float:
        """获取绳子张力（0-1）"""
        distance = np.linalg.norm(self.human.position - self.robot.position)
        return np.clip(distance / self.leash_length, 0, 1)

    def check_collision(
        self,
        obstacles,
        segment_obstacles=None,
        robot_radius: Optional[float] = None,
        human_radius: Optional[float] = None,
    ):
        """Check collision between robot/human and obstacles."""
        if (obstacles is None or len(obstacles) == 0) and (segment_obstacles is None or len(segment_obstacles) == 0):
            return False, None

        robot_radius = self.robot_radius if robot_radius is None else robot_radius
        human_radius = self.human_radius if human_radius is None else human_radius

        robot_pos = self.robot.position
        human_pos = self.human.position

        if obstacles is not None and len(obstacles) > 0:
            for idx, obs in enumerate(obstacles):
                if isinstance(obs, dict):
                    ox = float(obs.get("x", 0.0))
                    oy = float(obs.get("y", 0.0))
                    radius = float(obs.get("r", 0.0))
                else:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])

                dx_r = robot_pos[0] - ox
                dy_r = robot_pos[1] - oy
                if dx_r * dx_r + dy_r * dy_r <= (radius + robot_radius) ** 2:
                    return True, {
                        "type": "circle",
                        "who": "robot",
                        "idx": int(idx),
                        "obstacle": [ox, oy, radius],
                    }

                dx_h = human_pos[0] - ox
                dy_h = human_pos[1] - oy
                if dx_h * dx_h + dy_h * dy_h <= (radius + human_radius) ** 2:
                    return True, {
                        "type": "circle",
                        "who": "human",
                        "idx": int(idx),
                        "obstacle": [ox, oy, radius],
                    }

        if segment_obstacles is not None and len(segment_obstacles) > 0:
            def point_segment_dist_sq(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom < 1e-12:
                    diff = point - a
                    return float(np.dot(diff, diff))
                t = float(np.dot(point - a, ab)) / denom
                t = float(np.clip(t, 0.0, 1.0))
                closest = a + t * ab
                diff = point - closest
                return float(np.dot(diff, diff))

            for idx, seg in enumerate(segment_obstacles):
                if isinstance(seg, dict):
                    if "p1" in seg and "p2" in seg:
                        p1 = np.array(seg["p1"], dtype=np.float32)
                        p2 = np.array(seg["p2"], dtype=np.float32)
                    else:
                        p1 = np.array([seg.get("x1", 0.0), seg.get("y1", 0.0)], dtype=np.float32)
                        p2 = np.array([seg.get("x2", 0.0), seg.get("y2", 0.0)], dtype=np.float32)
                else:
                    p1 = np.array([seg[0], seg[1]], dtype=np.float32)
                    p2 = np.array([seg[2], seg[3]], dtype=np.float32)

                if point_segment_dist_sq(robot_pos, p1, p2) <= (robot_radius ** 2):
                    return True, {
                        "type": "segment",
                        "who": "robot",
                        "idx": int(idx),
                        "obstacle": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    }
                if point_segment_dist_sq(human_pos, p1, p2) <= (human_radius ** 2):
                    return True, {
                        "type": "segment",
                        "who": "human",
                        "idx": int(idx),
                        "obstacle": [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                    }

        return False, None


if __name__ == "__main__":
    # 测试物理引擎
    engine = PhysicsEngine()
    engine.reset(np.array([0.0, 0.0]))
    
    # 模拟前进
    engine.set_control(forward=1.0, turn=0.0)
    for i in range(100):
        robot, human = engine.step()
        if i % 20 == 0:
            print(f"Step {i}: Robot={robot.position}, Human={human.position}")
