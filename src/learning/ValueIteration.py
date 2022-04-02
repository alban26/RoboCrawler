import numpy as np
import logging
import time
from learning.LearningAlgorithm import LearningAlgorithm


class ValueIteration(LearningAlgorithm):
    """
    Value Iteration for reinforcement learning of the robot.
    """

    gamma = 0.9

    def __init__(self, myRobot, myWorld, ex):
        """
        :param myRobot: robot as RobotDiscrete
        :param myWorld: World
        """
        super().__init__(myRobot, myWorld, ex)
        logging.debug("ValueIteration init: ".format())

        # self.stopwatch = 0  # TODO: remove or needed?

    def reset(self):
        """
        Resets the learning process.
        :return:
        """
        super().reset()
        # value function, V(s) = 0 at the beginning
        self.value = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.policy = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.reward = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num + [self.myRobot.action_size()])

        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros(self.state.shape)

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, tricks, invert_learning, ui):
        """
        Learns the robot with Value Iteration.
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

            logging.debug("Value Table:\n{}".format(self.value))

            xi = np.random.random()
            if xi < self.epsilon:
                # exploration
                possible_actions = self.myRobot.get_possible_actions(self.state)
                a_forward = possible_actions[np.random.randint(0, len(possible_actions))]
            else:
                # exploitation
                a_forward = self.get_greedy_action(self.state)
            a_forward_diff = np.array(self.myRobot.action_to_diff[a_forward])

            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a_forward])

            # reward from world and some extra reward for not changing the joint direction => longer moves
            reward = self.myWorld.step_reward() + np.sum(np.clip(self.last_action_diff * a_forward_diff, 0, 1)) * 1.0
            if invert_learning:
                reward = -reward

            self.reward[tuple(self.state.flatten()) + (a_forward,)] = reward
            if tricks:
                # Trick 1, arms are symmetric (only useful for multiple arms):
                if self.myRobot.arms_num == 2:
                    state_mirrored = tuple(np.roll(self.state, 1, axis=0).flatten())
                    a_forward_mirrored = self.myRobot.get_action_of_diff(np.roll(a_forward_diff, 1, axis=0))
                    self.reward[state_mirrored + (a_forward_mirrored,)] = reward

                # Trick 2: reverse action has reward * (-1) (always useful):
                a_reverse = self.myRobot.get_action_reverse(a_forward)
                self.reward[tuple(successor_state.flatten()) + (a_reverse,)] = -reward
                if self.myRobot.arms_num == 2:
                    successor_state_mirrored = tuple(np.roll(successor_state, 1, axis=0).flatten())
                    a_reverse_mirrored = self.myRobot.get_action_of_diff(
                        np.roll(self.myRobot.action_to_diff[a_reverse], 1, axis=0))
                    self.reward[successor_state_mirrored + (a_reverse_mirrored,)] = -reward  # Trick 1

            if self.steps % improve_every_steps == 0:
                self.improve_value_and_policy()

            self.state = successor_state
            self.last_action_diff = a_forward_diff
            self.steps += 1

            self.ex.update_ui_step(self.steps, self.epsilon)

        self.ex.update_ui_finished()

        self.print_value_table()
        self.print_policy_table()

        return True

    def get_policy(self):
        return self.value, self.reward

    def set_policy(self, policy):
        (self.value, self.reward) = policy

    def get_table(self):
        return self.value

    def execute(self):
        """
        Executes learned policy in an endless loop.
        :return:
        """
        while not self.stop:
            while self.pause and not self.stop:
                time.sleep(0.1)
            a = self.get_greedy_action(self.state)
            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a])
            _ = self.myWorld.step_reward()
            self.state = successor_state

    def improve_value_and_policy(self):
        """
        Updates the value and policy table in each each state.
        :return:
        """
        for state in self.myRobot.get_all_states():
            a_value, a = self.max_long_term_reward(state)
            self.value[tuple(state.flatten())] = a_value
            self.policy[tuple(state.flatten())] = a

    def get_greedy_action(self, state):
        """
        Returns the action with the highest reward for the given state.
        :param state: state as tuple (state_arm1, state_arm2)
        :return: action as number
        """
        return self.max_long_term_reward(state)[1]

    def max_long_term_reward(self, state):
        """
        Calculates the next action with the highest discounted reward for the given state.
        :param state: state as tuple (state_arm1, state_arm2)
        :return: value of the action with the highest reward and the action itself as (value of action, action as number between 0 and 3)
        """
        max_a = None
        argmax_a = None
        for a in self.myRobot.get_possible_actions(state):
            long_term_reward = self.reward[tuple(state.flatten()) + (a,)] + self.gamma * self.value[
                tuple(self.myRobot.transition(state, a).flatten())]
            if max_a is None or long_term_reward > max_a:
                max_a = long_term_reward
                argmax_a = a
        return max_a, argmax_a

    def print_policy_table(self):
        """
        Prints the current policy table. (only for 2D policy table)
        :return:
        """
        if self.policy.ndim == 2:
            logging.debug("Policy:")
            for x in range(0, self.policy.shape[0]):
                line = ""
                for y in range(0, self.policy.shape[1]):
                    if self.policy[x, y] == 0:
                        line += "v"
                    elif self.policy[x, y] == 1:
                        line += "^"
                    elif self.policy[x, y] == 2:
                        line += ">"
                    elif self.policy[x, y] == 3:
                        line += "<"
                logging.debug(line)

    def print_value_table(self):
        """
        Prints the value table. (only for 2D value table)
        :return:
        """
        if self.value.ndim == 2:
            logging.debug("Value table:")
            for x in range(0, self.value.shape[0]):
                line = ""
                for y in range(0, self.value.shape[1]):
                    line += "   {0:.2f}".format(self.value[x, y])
                logging.debug(line)
