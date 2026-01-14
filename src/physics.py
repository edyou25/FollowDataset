"""
物理模型 - 机器人运动 + 绳子牵引人类跟随
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotState:
    """机器人状态"""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    heading: float = 0.0  # 朝向角度（弧度）
    
    def copy(self):
        return RobotState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=self.heading
        )


@dataclass
class HumanState:
    """人类状态"""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    
    def copy(self):
        return HumanState(
            position=self.position.copy(),
            velocity=self.velocity.copy()
        )


class PhysicsEngine:
    """物理引擎 - 处理机器人控制和人类跟随"""
    
    def __init__(
        self,
        leash_length: float = 1.5,  # 绳子长度（米）
        robot_speed: float = 1.5,   # 机器人移动速度（米/秒）
        turn_speed: float = 1.5,    # 机器人转向速度（弧度/秒，降低以匹配较慢的运动）
        human_drag: float = 0.9,    # 人类阻尼系数
        dt: float = 0.02            # 时间步长（秒）
    ):
        self.leash_length = leash_length
        self.robot_speed = robot_speed
        self.turn_speed = turn_speed
        self.human_drag = human_drag
        self.dt = dt
        
        # 状态
        self.robot = RobotState()
        self.human = HumanState()
        
        # 控制输入
        self.forward_input = 0.0  # -1 到 1
        self.turn_input = 0.0     # -1 到 1
    
    def reset(self, start_position: Optional[np.ndarray] = None):
        """重置物理状态"""
        if start_position is None:
            start_position = np.array([0.0, 0.0])
        
        self.robot = RobotState(
            position=start_position.copy(),
            velocity=np.array([0.0, 0.0]),
            heading=0.0
        )
        
        # 人在机器人后方
        human_offset = np.array([-self.leash_length * 0.8, 0.0])
        self.human = HumanState(
            position=start_position + human_offset,
            velocity=np.array([0.0, 0.0])
        )
        
        self.forward_input = 0.0
        self.turn_input = 0.0
    
    def set_control(self, forward: float, turn: float):
        """设置控制输入"""
        self.forward_input = np.clip(forward, -1.0, 1.0)
        self.turn_input = np.clip(turn, -1.0, 1.0)
    
    def step(self) -> tuple:
        """
        执行一步物理模拟
        
        Returns:
            tuple: (robot_state, human_state)
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
        self._update_human()
        
        return self.robot.copy(), self.human.copy()
    
    def _update_human(self):
        """更新人类位置 - 绳子牵引物理"""
        # 计算机器人到人的向量
        robot_to_human = self.human.position - self.robot.position
        distance = np.linalg.norm(robot_to_human)
        
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
                
            # 应用阻尼
            self.human.velocity = self.human.velocity * self.human_drag
            
            # 更新人的位置
            self.human.position = self.human.position + self.human.velocity * self.dt
            
            # 硬约束：确保不超过绳子长度
            robot_to_human = self.human.position - self.robot.position
            distance = np.linalg.norm(robot_to_human)
            if distance > self.leash_length:
                leash_dir = robot_to_human / distance
                self.human.position = self.robot.position + leash_dir * self.leash_length
    
    def get_leash_tension(self) -> float:
        """获取绳子张力（0-1）"""
        distance = np.linalg.norm(self.human.position - self.robot.position)
        return np.clip(distance / self.leash_length, 0, 1)


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

