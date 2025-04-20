import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from .reward import grasp_reward


class FR5_Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gui=False):
        super(FR5_Env).__init__()
        self.step_num = 0
        self.Con_cube = None

        # 动作空间（前6为关节，第7维为夹爪）
        low_action = np.array([-1.0]*6 + [0.0])
        high_action = np.array([1.0]*6 + [1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # 观测空间（+1维 Fx 夹爪总力）
        low = np.zeros((1, 13), dtype=np.float32)
        high = np.ones((1, 13), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 初始化 PyBullet
        if gui:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.init_env()

    def init_env(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fr5_path = os.path.join(base_dir, "fr5_description/urdf/fr5v6.urdf")

        self.fr5 = self.p.loadURDF(fr5_path, useFixedBase=True,
                                   basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
                                   flags=p.URDF_USE_SELF_COLLISION)

        self.table = self.p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        target_shape = self.p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=target_shape, basePosition=[0.5, 0.5, 2])
        self.p.changeDynamics(self.target, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.01)

        platform_shape = self.p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.3)
        self.targettable = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=platform_shape, basePosition=[0.5, 0.5, 2])

        self.p.changeDynamics(self.fr5, 8, lateralFriction=1.0)
        self.p.changeDynamics(self.fr5, 9, lateralFriction=1.0)

        # 启用夹爪力传感器
        self.p.enableJointForceTorqueSensor(self.fr5, 8, enableSensor=True)
        self.p.enableJointForceTorqueSensor(self.fr5, 9, enableSensor=True)

        # 创建障碍物（两个红色盒子）
        self.obstacle_ids = []
        for i in range(2):
            obs_shape = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
            obs_body = self.p.createMultiBody(baseMass=0,
                                              baseCollisionShapeIndex=obs_shape,
                                              basePosition=[0.3 + 0.1*i, 0.65, 0.05])
            self.p.changeVisualShape(obs_body, -1, rgbaColor=[1, 0, 0, 1])
            self.obstacle_ids.append(obs_body)

    def step(self, action):
        info = {}
        joint_angles = [p.getJointState(self.fr5, i)[0] for i in [1,2,3,4,5,6]]
        target_angles = np.array(joint_angles) + (np.array(action[0:6]) / 180 * np.pi)

        # 控制夹爪闭合程度（0.0=张开，1.0=闭合）
        grip_cmd = action[6]
        grip_pos = 0.04 * (1.0 - grip_cmd)

        p.setJointMotorControlArray(self.fr5, [1,2,3,4,5,6], p.POSITION_CONTROL, targetPositions=target_angles)
        p.setJointMotorControlArray(self.fr5, [8,9], p.POSITION_CONTROL, targetPositions=[grip_pos, grip_pos])

        for _ in range(20):
            self.p.stepSimulation()

        self.reward, info = grasp_reward(self)
        self.get_observation()
        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False

        neutral_angle = [-49.46, -57.60, -138.39, -164.00, -49.46, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        p.setJointMotorControlArray(self.fr5, [1,2,3,4,5,6,8,9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        # 目标位置
        self.goalx = np.random.uniform(-0.2, 0.2)
        self.goaly = np.random.uniform(0.6, 0.8)
        self.goalz = np.random.uniform(0.1, 0.3)
        self.target_position = [self.goalx, self.goaly, self.goalz]
        self.targettable_position = [self.goalx, self.goaly, self.goalz - 0.175]
        self.p.resetBasePositionAndOrientation(self.targettable, self.targettable_position, [0,0,0,1])
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0,0,0,1])

        # 生成合理障碍物位置
        def generate_valid_positions(forbidden_zones, num=2, min_dist=0.1, max_trials=100):
            positions = []
            trials = 0
            while len(positions) < num and trials < max_trials:
                x = np.random.uniform(-0.2, 0.2)
                y = np.random.uniform(0.6, 0.8)
                candidate = np.array([x, y])
                valid = True
                for zone in forbidden_zones + positions:
                    if np.linalg.norm(candidate - np.array(zone)) < min_dist:
                        valid = False
                        break
                if valid:
                    positions.append((x, y))
                trials += 1
            return positions

        forbidden_positions = [(0, 0.5), (self.goalx, self.goaly)]
        obstacle_xy = generate_valid_positions(forbidden_positions, num=2, min_dist=0.12)
        for i, (x, y) in enumerate(obstacle_xy):
            self.p.resetBasePositionAndOrientation(self.obstacle_ids[i], [x, y, 0.05], [0,0,0,1])

        for _ in range(100):
            self.p.stepSimulation()

        self.get_observation()

        return self.observation, {
            'is_success': False,
            'reward': 0,
            'step_num': 0
        }

    def get_observation(self, add_noise=False):
        tip_pos = np.array(p.getLinkState(self.fr5, 6)[0])
        ori = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        gripper_pos = tip_pos + ori.apply(np.array([0, 0, 0.15]))

        joint_angles = []
        for i in range(1, 7):
            angle = p.getJointState(self.fr5, i)[0] * 180 / np.pi
            joint_angles.append(self.add_noise(angle, 0, True) if add_noise else angle)

        force_8 = self.p.getJointState(self.fr5, 8)[2]
        force_9 = self.p.getJointState(self.fr5, 9)[2]
        grip_force_x = np.abs(force_8[0]) + np.abs(force_9[0])

        obs = np.hstack((
            ((gripper_pos + [0.922, 0.922, 0.5]) / [1.844, 1.844, 1.0]),
            ((np.array(joint_angles) / 180 + 1) / 2),
            ((np.array(self.target_position) + [0.2, -0.6, -0.1]) / [0.4, 0.2, 0.2]),
            grip_force_x / 50.0
        )).astype(np.float32).reshape(1, -1)

        self.observation = obs

    def render(self):
        self.p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=180, cameraPitch=0,
                                          cameraTargetPosition=[0.39, 0.45, 0.42])

    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        if gaussian:
            return angle + np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            return angle + random.uniform(-5, 5)


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = FR5_Env(gui=True)
    env.reset()
    check_env(env, warn=True)
    env.render()
    print("test going")
    time.sleep(10)
    time.sleep(100)

