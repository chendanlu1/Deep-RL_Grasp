import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from loguru import logger


def judge_success(self, height_threshold=0.4):
    """视觉抓取成功判定（抬升高度）"""
    target_pos = p.getBasePositionAndOrientation(self.target)[0]
    self.success = target_pos[2] > height_threshold


def cal_success_reward(self, distance):
    """成功加分 + 防止非法接触"""
    reward = 0

    obstacle_collision = any(
        len(p.getContactPoints(bodyA=self.fr5, bodyB=obs_id)) > 0
        for obs_id in getattr(self, 'obstacle_ids', [])
    )

    if self.success and self.step_num <= 100:
        reward = 1000
        self.terminated = True
        logger.info(f"[Success] Object lifted at step {self.step_num}, distance: {distance:.4f}")

    elif obstacle_collision:
        reward = -100
        self.terminated = True
        logger.info(f"[Failure] Collision with obstacle at step {self.step_num}")

    elif self.step_num > 100:
        reward = -100
        self.terminated = True
        logger.info(f"[Failure] Step limit exceeded: {self.step_num}")

    return reward


def cal_dis_reward(self, distance):
    """视觉主导的靠近奖励"""
    if self.step_num == 0:
        reward = 0
    else:
        reward = 150 * (self.distance_last - distance)
    self.distance_last = distance
    return reward


def cal_pose_reward(self):
    """鼓励末端竖直下抓"""
    gripper_orientation = p.getLinkState(self.fr5, 7)[1]
    euler = R.from_quat(gripper_orientation).as_euler('xyz', degrees=True)
    pose_reward = -(pow(euler[0] + 90, 2) + pow(euler[1], 2) + pow(euler[2], 2))
    return pose_reward * 0.1


def cal_grip_force_reward(self):
    """夹力范围控制"""
    force_left = np.linalg.norm(p.getJointState(self.fr5, 8)[2])
    force_right = np.linalg.norm(p.getJointState(self.fr5, 9)[2])
    total_force = force_left + force_right

    if 2.0 < total_force < 5.0:
        return 10.0
    elif total_force >= 5.0:
        return -10.0
    elif total_force <= 2.0:
        return -5.0
    return 0.0


def get_distance(self):
    """抓取偏差距离：末端中心 vs 目标中心"""
    gripper_tip_pos = np.array(p.getLinkState(self.fr5, 6)[0])
    orientation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
    offset = np.array([0, 0, 0.15])
    gripper_center = gripper_tip_pos + orientation.apply(offset)
    target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0])
    return np.linalg.norm(gripper_center - target_pos)


def grasp_reward(self):
    """总奖励函数：视觉靠近 + 成功抓取为主，辅以姿态/夹力/效率"""
    info = {}
    distance = self.get_distance_from_vision()   
    judge_success(self)

    success_reward = cal_success_reward(self, distance)
    distance_reward = cal_dis_reward(self, distance)
    pose_reward = cal_pose_reward(self)
    grip_force_reward = cal_grip_force_reward(self)
    efficiency_penalty = -0.05 * self.step_num

    total_reward = (
        success_reward +
        distance_reward +
        pose_reward +
        grip_force_reward +
        efficiency_penalty
    )

    self.truncated = False
    self.reward = total_reward

    info['reward'] = total_reward
    info['is_success'] = self.success
    info['step_num'] = self.step_num
    info['success_reward'] = int(self.success)
    info['distance_reward'] = distance_reward
    info['pose_reward'] = pose_reward
    info['grip_force_reward'] = grip_force_reward
    info['efficiency_penalty'] = efficiency_penalty

    return total_reward, info

