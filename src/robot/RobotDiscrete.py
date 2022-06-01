import itertools
import logging
import random

import numpy as np

from robot.Robot import Robot, Mode


class RobotDiscrete(Robot):
    """
    Robot with discrete states and actions.
    """

    def __init__(self):
        pass

    def init(self, mode, arms_num, joints_per_arm_num, state_range_deg_joints, joints_states_num):
        """
        :param mode: Mode enum
        :param arms_num: number of arms
        :param joints_per_arm_num: number of joints per arm
        :param state_range_deg_joints: array of range of each joint in degree,
            for example [[-20, 40], [40, 140]] where [[joint1], [joint2]]
        :param joints_states_num: number of states or each joint, array, for example [5, 5]
            where [states_joint1, states_joint2]
        :return:
        """
        super().__init__(mode, arms_num, joints_per_arm_num, state_range_deg_joints)

        self.joints_states_num = joints_states_num

        # mapping from arm state to respective degree
        self.state_lookuptable_joints = []
        for i, state_range_deg_joint in enumerate(self.state_range_deg_joints):
            self.state_lookuptable_joints.append(
                np.deg2rad(np.linspace(state_range_deg_joint[0], state_range_deg_joint[1],
                                       num=self.joints_states_num[i])))

        # calculate all states
        range_per_joint = []
        for joint in self.arms_num * self.joints_states_num:
            range_per_joint.append(list(range(joint)))
        self.all_states = np.array([i for i in itertools.product(*range_per_joint)]) \
            .reshape(-1, self.arms_num, self.joints_per_arm_num)

        # mapping from a state to a statenr (all states are numbered)
        self.state_to_statenr = {}
        for i, state in enumerate(self.all_states):
            self.state_to_statenr[tuple([tuple(j) for j in state])] = i
        self.statenr_to_state = {v: k for k, v in self.state_to_statenr.items()}
        logging.debug(len(self.statenr_to_state))

        # calculate all actions (number for each action) and diff (action difference in tuple)
        self.diff_to_action = {}  # mapping from action as diff to actions as number
        if self.mode is Mode.ONE_ACTION:
            for j in range(self.joints_per_arm_num):
                diff_zero = (0,) * self.joints_per_arm_num
                diff_pos = [0] * (self.joints_per_arm_num - 1)
                diff_neg = [0] * (self.joints_per_arm_num - 1)
                diff_pos.insert(j, 1)
                diff_neg.insert(j, -1)
                diff_pos = tuple(diff_pos)
                diff_neg = tuple(diff_neg)
                for i in range(self.arms_num):
                    diff_zeros = [diff_zero] * (self.arms_num - 1)
                    c1 = diff_zeros.copy()
                    c1.insert(i, diff_pos)
                    c2 = diff_zeros.copy()
                    c2.insert(i, diff_neg)
                    self.diff_to_action[tuple(c1)] = 2 * (j * self.arms_num + i)
                    self.diff_to_action[tuple(c2)] = 2 * (j * self.arms_num + i) + 1

        if self.mode is Mode.ALL_POSSIBLE:
            actions_per_arm = []
            a = [0, 1, -1]
            b = [a] * self.joints_per_arm_num
            for e in itertools.product(*b):
                actions_per_arm.append(e)
            list_for_product = self.arms_num * [actions_per_arm]
            for i, diff in enumerate(itertools.product(*list_for_product)):
                self.diff_to_action[diff] = i

        # mapping from action as number to action as difference
        self.action_to_diff = {v: k for k, v in self.diff_to_action.items()}
        logging.debug(len(self.action_to_diff))

        self.possible_actions_per_state = {}
        for state in self.all_states:
            possible_actions = []
            for action in self.action_to_diff.keys():
                next_state = self.transition(state, action)
                if np.array(np.where(next_state < 0)).size == 0:  # next state smaller than zero?
                    broken = False
                    for i, joint_states_num in enumerate(self.joints_states_num):
                        # next state larger than allowed?
                        if np.array(np.where(next_state[:, i] >= joint_states_num)).size > 0:
                            broken = True
                            break
                    if not broken:
                        possible_actions.append(action)
            self.possible_actions_per_state[tuple([tuple(j) for j in state])] = np.array(possible_actions)

        self.possible_action_diffs_per_state = {}
        for state in self.all_states:
            possible_actions = []
            for action in self.possible_actions_per_state[tuple([tuple(j) for j in state])]:
                possible_actions.append(self.action_to_diff[action])
            self.possible_action_diffs_per_state[tuple([tuple(j) for j in state])] = np.array(possible_actions)

        # set initial state
        self.state = []  # [[2, 2], [2, 2]]  # [[arm1_joint1, arm1_joint2], [arm2_joint1, arm2_joint2]]
        for arm in range(self.arms_num):
            arm = []
            for joint_states in self.joints_states_num:
                arm += [int(joint_states / 2)]  # start each joint in the middle of the states
            self.state.append(arm)
        self.state = np.array(self.state)

        logging.debug("RobotDiscrete init: joints_states_num={}, action size={}, state size={}, inital state={}".format(
            self.joints_states_num, len(self.diff_to_action.keys()), len(self.state_to_statenr.keys()),
            self.state))

    def apply_action(self, action):
        """
        Apply the given action (arm1_change, ..., armN_change) to the robots state.
        :param action: action in diff format (arm1_change, arm2_change, armN_change)
        :return: the robot state after the action
        """
        action = np.array(action)
        if self.mode is Mode.ONE_ACTION:
            if abs(np.sum(action)) != 1:
                logging.error("Action is not valid!")
            arm = np.array(np.where(np.abs(np.sum(action, 1))))[0][0]
            logging.debug("Moved arm {} from state {} to {}".format(arm, self.state[arm], (self.state + action)[arm]))
        elif self.diff_to_action[tuple([tuple(j) for j in action])] not in self.get_possible_actions(self.state):
            logging.error("Action is not valid!")
            return self.state.copy()
        else:
            pass
            logging.debug("Moved from state {} to {}".format(self.state, (self.state + action)))
        self.state += action
        return self.state.copy()

    def get_statenr_of_state(self, state):
        """
        Get state as state number
        :return:
        """
        return self.state_to_statenr[tuple([tuple(j) for j in state])]

    def get_all_states(self):
        """
        Get all states as an array of arrays.
        :return: states as array of arrays.
        """
        return self.all_states

    def get_statenr(self):
        """
        Get current state as state number
        :return:
        """
        return self.state_to_statenr[tuple([tuple(j) for j in self.get_state()])]

    def get_action_of_diff(self, diff):
        return self.diff_to_action[tuple([tuple(j) for j in diff])]

    def get_possible_actions(self, state):
        """
        Returns all possible next actions given the current state.
        :param state: state as array [[arm1_joint1, arm1_joint2], [arm2_joint1, arm2_joint2]]
        :return: list of actions as number each
        """
        return self.possible_actions_per_state[tuple([tuple(j) for j in state])]

    def get_possible_action_diffs(self, state):
        """
        Returns all possible next actions given the current state.
        :param state: state as array [[arm1_joint1, arm1_joint2], [arm2_joint1, arm2_joint2]]
        :return: list of actions as diff each
        """
        return self.possible_action_diffs_per_state[tuple([tuple(j) for j in state])]

    def get_action_reverse(self, action):
        """
        Returns the reverse action of the given action.
        :param action: action as number
        :return: action as number
        """
        return self.diff_to_action[tuple([tuple(j) for j in -np.array(self.action_to_diff[action])])]

    def transition(self, state, action):
        """
        Returns next state s' given action a and state s.
        :param state: state as array [[arm1_joint1, arm1_joint2], [arm2_joint1, arm2_joint2]]
        :param action: action as number
        :return:
        """
        diff = self.action_to_diff[action]
        return state + np.array(diff)

    def action_size(self):
        """
        Returns number of actions
        :return: number of actions
        """
        return len(self.diff_to_action.keys())

    def state_size(self):
        """
        Returns number of states
        :return: number of states
        """
        return len(self.state_to_statenr.keys())

    def get_state(self):
        """
        Get current state
        :return: current robot state
        """
        return self.state.copy()

    def state_shape(self):
        """
        Get shape of state
        :return: shape of state
        """
        return self.state.shape

    def get_all_possible_state_action(self):
        state_action_pairs = []
        for state in self.get_all_states():
            for action in self.get_possible_actions(state):
                state_action_pairs.append((state, action))
        random.shuffle(state_action_pairs)
        return state_action_pairs

    def get_arm_states_degree(self):
        """
        Returns degrees of robot arm (degree_arm1, ..., degree_arm4).
        Works only for one for robot with two arms and two joints for each arm
        :return: (degree_arm1, ..., degree_arm4)
        """
        if np.shape(self.state) == (2, 2):
            degree1 = np.rad2deg(self.state_lookuptable_joints[0][self.state[0, 0]])
            degree2 = np.rad2deg(self.state_lookuptable_joints[1][self.state[0, 1]])
            degree3 = np.rad2deg(self.state_lookuptable_joints[0][self.state[1, 0]])
            degree4 = np.rad2deg(self.state_lookuptable_joints[1][self.state[1, 1]])
        else:
            degree1 = np.rad2deg(self.state_lookuptable_joints[0][self.state[0, 0]])
            degree2 = np.rad2deg(self.state_lookuptable_joints[1][self.state[0, 1]])
            degree3 = degree1
            degree4 = degree2

        return degree1, degree2, degree3, degree4
