import numpy as np
import random
import logging
import time
from learning.LearningAlgorithm import LearningAlgorithm


class QLearning(LearningAlgorithm):
    """
    QLearning
    """

    gamma = 0.95
    learning_rate = 1.0

    def __init__(self, myRobot, myWorld, ex):
        super().__init__(myRobot, myWorld, ex)

        logging.debug("QLearning init: ".format())

    def reset(self):
        """
        Resets the learning process.
        :return:
        """
        super().reset()
        self.qtable = np.zeros((self.myRobot.state_size(), self.myRobot.action_size()))
        self.reward_cap = 1

        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros((self.myRobot.joints_per_arm_num))

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):
        """
        Learns the robot with QLearning.
        :param steps: number of iterations/steps
        :param min_epsilon
        :param max_epsilon: bracket around possible epsilon values for xi-decision
        :return: Returns bool if learning has finished or stopped
        """

        for _ in range(steps):
            if self.stop:
                return False
            while self.pause:
                self.myWorld.step_reward()
                time.sleep(0.1)

            logging.debug("Step: {}".format(self.steps))

            self.epsilon = (1 - (self.steps / steps)) * (max_epsilon - min_epsilon) + min_epsilon
            logging.debug("Epsilon: {}".format(self.epsilon))

            logging.debug("Q Table:\n{}".format(self.qtable))

            statenr = self.myRobot.get_statenr()
            """
            Holen des aktuellen Zustands
            """
            self.state = self.myRobot.get_state()
            xi = np.random.random()
            """
            Epsilon Greedy-Strategie
            """
            if xi < self.epsilon or np.max(self.qtable[statenr, :]) <= self.reward_cap:
                # exploration
                possible_actions = self.myRobot.get_possible_actions(self.state)
                a_forward = possible_actions[np.random.randint(0, len(possible_actions))]
            else:
                # exploitation
                a_forward = np.argmax(self.qtable[statenr, :])
            a_forward_diff = np.array(self.myRobot.action_to_diff[a_forward])
            """
            Ausführen der ermittelten Aktion
            + Holen des Nachfolgezustands
            """
            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a_forward])
            successor_state_nr = self.myRobot.get_statenr_of_state(successor_state)

            # reward from world and some extra reward for not changing the joint direction => longer moves
            """
            Holen der Belohnung
            """
            reward = self.myWorld.step_reward() # + np.sum(np.clip(self.last_action_diff * a_forward_diff, 0, 1)) * 1.0
            if invert_learning:
                reward = -reward

            # Q learning function for deterministic environment
            self.qtable[statenr, a_forward] = reward + self.gamma * np.max(self.qtable[successor_state_nr, :])
            logging.debug("Statenr: {}, a_forward: {}, successor_state: {}".format(statenr, a_forward, successor_state_nr))

            """
            TD-Lernen
            """
            # Q learning function for non deterministic environment
            # self.qtable[statenr, a_forward] = self.qtable[statenr, a_forward] + self.learning_rate * (
            #    reward + self.gamma * np.max(self.qtable[successor_state, :]) - self.qtable[statenr, a_forward])

            # if tricks:
            #     # Trick 1, arms are symmetric (only useful for multiple arms):
            #     if self.myRobot.arms_num == 2:
            #         statenr_mirrored = self.myRobot.get_statenr_of_state(np.roll(self.state, 1, axis=0))
            #         a_forward_mirrored = self.myRobot.get_action_of_diff(np.roll(a_forward_diff, 1, axis=0))
            #         self.qtable[statenr_mirrored, a_forward_mirrored] = reward + self.gamma * np.max(self.qtable[successor_state_nr, :]) # reward + self.gamma * np.max(self.qtable[self.myRobot.get_statenr_of_state(self.myRobot.transition(self.state, a_forward)), :])
            #
            #     # Trick 2: reverse action has reward * (-1) (always useful):
            #     a_reverse = self.myRobot.get_action_reverse(a_forward)
            #     self.qtable[self.myRobot.get_statenr_of_state(successor_state), a_reverse] = -reward + self.gamma * np.max(self.qtable[statenr, :])
            #     if self.myRobot.arms_num == 2:
            #         successor_state_mirrored = self.myRobot.get_statenr_of_state(np.roll(successor_state, 1, axis=0))
            #         a_reverse_mirrored = self.myRobot.get_action_of_diff(np.roll(self.myRobot.action_to_diff[a_reverse], 1, axis=0))
            #         self.qtable[successor_state_mirrored, a_reverse_mirrored] = -reward + self.gamma * np.max(self.qtable[statenr, :])  # Trick 1

            # possibility to improve tables only every few steps
            # if self.steps % improve_every_steps == 0:
            #     self.improve_value_and_policy()
            """
            Setzen des neuen Zustandes
            """
            self.state = successor_state
            self.last_action_diff = a_forward_diff
            self.steps += 1

            """
            Vorgehen: 
            - Erfahrungen sammeln (Replay Puffer auffüllen) (Zustand, Aktion, Belohnung, Folgezustand)
            - sobald Puffer gewisse Größe erreicht hat mit den 
              gesammelten Erfahrungen Netz Trainieren (2 Netze, 1 online-Netz und 1 Target-Netz)
            - ...
            """

            self.ex.update_ui_step(self.steps, self.epsilon)

        self.ex.update_ui_finished()
        self.print_q_table()

        return True

    def get_policy(self):
        return (self.qtable,)

    def set_policy(self, policy):
        (self.qtable,) = policy

    def get_table(self):
        return self.qtable

    def execute(self):
        """
        Executes learned policy in an endless loop.
        :return:
        """
        while not self.stop:
            while self.pause and not self.stop:
                time.sleep(0.1)
            state = self.myRobot.get_state()
            statenr = self.myRobot.get_statenr()
            if np.max(self.qtable[statenr, :]) <= self.reward_cap:
                possible_actions = self.myRobot.get_possible_actions(state)
                a = possible_actions[np.random.randint(0, len(possible_actions))]
            else:
                a = np.argmax(self.qtable[statenr, :])
            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a])
            _ = self.myWorld.step_reward()

    def print_q_table(self):
        """
        Prints the Q table. (only for 2D value table)
        :return:
        """
        if self.qtable.ndim == 2:
            logging.debug("Q table:")
            for x in range(0, self.qtable.shape[0]):
                line = ""
                for y in range(0, self.qtable.shape[1]):
                    line += "   {0:.2f}".format(self.qtable[x, y])
                logging.debug(line)