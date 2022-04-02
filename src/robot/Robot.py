import logging
import numpy as np
from enum import Enum


class Mode(Enum):
    ONE_ACTION = 0  # only one action in each step allowed
    ONE_ACTION_PER_ARM = 1  # one action per arm in each step
    ALL_POSSIBLE = 2  # all joints are allowed to move in one step


class Robot:
    """
    The robot parent class with mode, arms and joints parameters.
    """

    def __init__(self, mode, arms_num, joints_per_arm_num, state_range_deg_joints):
        """
        :param mode: Mode enum
        :param arms_num: number of arms
        :param joints_per_arm_num: number of joints per arm
        :param state_range_deg_joints: array of range of each joint in degree,
            for example [[-20, 40], [40, 140]] where [[joint1], [joint2]]
        """
        self.mode = mode
        self.arms_num = arms_num
        self.joints_per_arm_num = joints_per_arm_num
        assert (len(state_range_deg_joints) == joints_per_arm_num)
        self.state_range_deg_joints = state_range_deg_joints
        self.state_range_rad_joints = np.deg2rad(np.array(self.state_range_deg_joints))

        logging.debug("Robot init: mode={}, arms_num={}, joints_per_arm_num={}, state_range_deg_joints={}".format(
            self.mode, self.arms_num, self.joints_per_arm_num, self.state_range_deg_joints,
            self.state_range_rad_joints))

    @staticmethod
    def diff(theta_0, theta_1):
        """
        Calculates angle between two angles.
        :param theta_0: first angle
        :param theta_1: second angle
        :return: difference of theta_0 and theta_1
        """
        return ((theta_0 - theta_1 + np.pi) % (2 * np.pi)) - np.pi

    def apply_action(self, action):
        """
        Apply the given action to the robot.
        :param action: action
        :return: the robot state after the action
        """
        return NotImplementedError
