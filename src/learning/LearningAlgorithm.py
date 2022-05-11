import pickle
import numpy as np
import torch


class LearningAlgorithm:
    """
    Interface/parent class for learning algorithms
    """

    def __init__(self, myRobot, myWorld, ex):
        """
        :param myRobot: robot instance
        :param myWorld: world instance
        :param ex: gui instance
        """
        self.myRobot = myRobot
        self.myWorld = myWorld
        self.ex = ex

        self.reset()

    def reset(self):
        """
        Resets the learning process.
        :return:
        """
        # Flags to stop and pause learning (in thread)
        self.stop = False
        self.pause = False

        self.steps = 0
        self.table_sum_array = []

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):
        """
        Learns the robot with Value Iteration.
        :param steps: number of iterations/steps
        :param min_epsilon
        :param max_epsilon: bracket around possible epsilon values for xi-decision
        :return: Returns bool if learning has finished or stopped
        """
        raise NotImplementedError("Must override learn")

    def get_pause(self):
        return self.pause

    def set_pause(self, val=True):
        self.pause = val

    def set_stop(self, val=True):
        self.stop = val

    def get_policy(self):
        raise NotImplementedError("Must override get_policy")

    def save_policy(self, file_path, param_vector):
        pickle.dump((self.get_policy(), param_vector), open(file_path, "wb"))

    def set_policy(self, policy):
        raise NotImplementedError("Must override set_policy")

    def load_policy(self, file_path):
        self.set_policy(pickle.load(open(file_path, "rb"))[0])

    def get_table(self):
        raise NotImplementedError("Must override get_table")

    def sum_table(self):
        np.sum(self.get_table())
        self.table_sum_array.append(np.sum(self.get_table()))
        return self.table_sum_array

    def execute(self):
        """
        Executes learned policy in an endless loop.

        :return:
        """
        raise NotImplementedError("Must override execute")
