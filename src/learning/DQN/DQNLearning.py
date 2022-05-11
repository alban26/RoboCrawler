import logging

import numpy as np
import torch
from torch import optim

from learning.LearningAlgorithm import LearningAlgorithm

from .DeepQNetwork import DQN
from .ReplayBuffer import ReplayBuffer, Experience


class DQNLearning(LearningAlgorithm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, myRobot, myWorld, ex):
        super().__init__(myRobot, myWorld, ex)

        self.memory = None
        self.last_action_diff = None
        self.state = None
        self.target_network = None
        self.policy_network = None
        self.LR = None
        self.NUM_EPISODES = None
        self.MEMORY_SIZE = None
        self.TARGET_UPDATE = None
        self.EPS_DECAY = None
        self.EPS_END = None
        self.EPS_START = None
        self.GAMMA = None
        self.BATCH_SIZE = None
        logging.debug("DQN-Learning init: ".format())

    def reset(self):
        """
        Resets the learning process.
        :return:
        """
        super().reset()
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        # Epsilon Greedy Parameter
        self.EPS_START = 1
        self.EPS_END = 0.01
        self.EPS_DECAY = 0.01

        # Wie oft soll das target Network aktualisiert werden
        self.TARGET_UPDATE = 10

        self.MEMORY_SIZE = 100000
        self.LR = 0.001
        self.NUM_EPISODES = 5000

        input_size = self.myRobot.arms_num * self.myRobot.joints_per_arm_num
        output_size = self.myRobot.action_size()

        self.policy_network = DQN(input_size, output_size)
        self.target_network = DQN(input_size, output_size)

        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros(self.myRobot.joints_per_arm_num)

        self.memory = ReplayBuffer(capacity=self.MEMORY_SIZE)
        optimizer = optim.Adam(params=self.policy_network.parameters(), lr=self.LR)

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):
        pass

    def get_policy(self):
        pass

    def set_policy(self, policy):
        pass

    def get_table(self):
        pass

    def execute(self):
        pass
