import logging
import pickle
import random
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from learning.LearningAlgorithm import LearningAlgorithm

from .DeepQNetwork import DQN
from .ReplayBuffer import ReplayBuffer, Experience


class DQNLearning(LearningAlgorithm):

    def __init__(self, myRobot, myWorld, ex):
        super().__init__(myRobot, myWorld, ex)

        self.device = None
        self.memory = None
        self.last_action_diff = None
        self.state = None
        self.target_network = None
        self.policy_network = None
        self.LR = None
        self.NUM_EPISODES = None
        self.STEPS_PER_EPISODE = None
        self.MEMORY_SIZE = None
        self.TARGET_UPDATE = None
        self.EPS_DECAY = None
        self.EPS_END = None
        self.EPS_START = None
        self.GAMMA = None
        self.BATCH_SIZE = None
        self.TAU = None
        self.optimizer = None
        self.time_step = None
        logging.debug("DQN-Learning init: ".format())

    def reset(self):
        """
        Resets the learning process.
        :return:
        """
        super().reset()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.TAU = 0.1
        # Epsilon Greedy Parameter
        self.EPS_START = 1
        self.EPS_END = 0.01
        self.EPS_DECAY = 0.995

        # Wie oft soll das target Network aktualisiert werden
        self.TARGET_UPDATE = 10

        self.MEMORY_SIZE = 100000
        self.LR = 0.001
        self.NUM_EPISODES = 16000
        self.STEPS_PER_EPISODE = 50

        input_size = self.myRobot.arms_num * self.myRobot.joints_per_arm_num
        output_size = self.myRobot.action_size()

        self.policy_network = DQN(input_size, output_size).to(device=self.device)
        self.target_network = DQN(input_size, output_size).to(device=self.device)

        self.target_network.eval()

        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros(self.myRobot.joints_per_arm_num)

        self.memory = ReplayBuffer(capacity=self.MEMORY_SIZE)
        self.optimizer = optim.Adam(params=self.policy_network.parameters(), lr=self.LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.time_step = 0

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):
        # epsilon = self.EPS_START
        # for episode in range(self.NUM_EPISODES):
        #     state = self.myRobot.get_state()
        #     for step in range(self.STEPS_PER_EPISODE):
        #         action = self.select_action(state=state, epsilon=epsilon)
        #         next_state = self.myRobot.apply_action(self.myRobot.action_to_diff[action.item()])
        #         reward = self.myWorld.step_reward()
        #         self.memory.append(Experience(state, action, reward, next_state))
        #         self.time_step = (self.time_step + 1)
        #         if self.time_step % self.TARGET_UPDATE == 0:
        #             # If enough samples are available in memory, get random subset and learn
        #             if len(self.memory) > self.BATCH_SIZE:
        #                 experiences = self.memory.sample(self.BATCH_SIZE)
        #                 self.train(experiences)
        #
        #     epsilon = max(self.EPS_END, self.EPS_DECAY * epsilon)
        #     self.ex.update_ui_step(episode, epsilon)
        #
        # save_network_string = datetime.now().strftime("%M_%S_%MS")
        # pickle.dump(self.policy_network,
        #             open(f"../neural_networks/{save_network_string}-610_109-Schritt-04-150.pkl", "wb"))
        self.policy_network = self.load_neural_network()
        self.ex.update_ui_finished()
        return True

    def load_neural_network(self):
        neural_network = None
        files = [f for f in listdir("../neural_networks") if isfile(join("../neural_networks", f))]
        for file in files:
            opentxt = f"../neural_networks/{file}"
            neural_network = pickle.load(open(opentxt, "rb"))
        return neural_network

    def select_action(self, state, epsilon=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if np.random.rand() > epsilon:
            # state = torch.tensor(state, device=self.device)
            state = torch.tensor(state.reshape(1, 4), dtype=torch.float32)
            self.policy_network.eval()  # Sets the module in evaluation mode.
            with torch.no_grad():
                action_values = self.policy_network(state)
            self.policy_network.train()  # Sets the module in training mode.
            return action_values.detach().argmax()
        else:
            return np.random.choice(self.myRobot.get_possible_actions(state))

    def train(self, experiences):
        """Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states = self.tensornize(experiences)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.GAMMA * Q_targets_next)

        # Get expected Q values from local model
        Q_expected = self.policy_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update()

    def tensornize(self, experiences):
        states, actions, rewards, next_states = experiences
        return torch.tensor(states.reshape(-1, 4), dtype=torch.float32), torch.tensor(actions,
                                                                                      dtype=torch.int64).unsqueeze(1), \
               torch.tensor(rewards).unsqueeze(1), torch.tensor(states.reshape(-1, 4), dtype=torch.float32)

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def get_policy(self):
        pass

    def set_policy(self, policy):
        pass

    def get_table(self):
        pass

    def execute(self):
        while not self.stop:
            while self.pause and not self.stop:
                time.sleep(0.1)
            a = self.select_action(self.state, epsilon=0.0)
            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a.item()])
            rew = self.myWorld.step_reward()
            # print(rew)
            self.state = successor_state
        # self.myWorld.draw_steps()
        # self.myWorld.draw_angles()
