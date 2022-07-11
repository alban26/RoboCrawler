import logging
import pickle
import random
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from timeit import default_timer as timer

from learning.LearningAlgorithm import LearningAlgorithm


class ValueIteration(LearningAlgorithm):
    GAMMA = 0.9

    load = True

    def __init__(self, myRobot, myWorld, ex):

        super().__init__(myRobot, myWorld, ex)
        self.reward = None
        self.execute_reward = None
        self.policy = None
        self.value = None
        self.last_action_diff = None
        self.total_reward = 0
        self.steps = 0

    def reset(self):

        super().reset()

        self.value = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.policy = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.reward = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num + [self.myRobot.action_size()])
        self.execute_reward = np.zeros(
            self.myRobot.joints_states_num * self.myRobot.arms_num + [self.myRobot.action_size()])
        self.std_dev = None
        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros(self.state.shape)

    def calc_reward(self):
        all_possible_state_action_pairs = self.myRobot.get_all_possible_state_action().copy()
        reward_tables = []
        k = 1
        for _ in range(k):
            n = 0
            N = len(all_possible_state_action_pairs)
            for state_action in all_possible_state_action_pairs:
                self.myRobot.state = state_action[0].copy()
                self.state = state_action[0].copy()
                self.myWorld.reset_sim()
                action_diff = self.myRobot.action_to_diff[state_action[1]]

                next_state = self.myRobot.apply_action(action_diff)

                action_diff = np.array(action_diff)
                reward = self.myWorld.step_reward()

                self.reward[tuple(state_action[0].flatten()) + (state_action[1],)] = reward

                self.last_action_diff = action_diff
                n += 1
                if n % 1000 == 0:
                    print('\rIteration {}/{}'.format(n, N), end="")

            reward_tables.append(self.reward)
            print(reward_tables)

        # Mittelt die Reward Tabellen
        reward_mean = np.zeros(self.reward.shape)
        for rt in reward_tables:
            reward_mean += rt
        self.reward = reward_mean / k

        save_reward_string = datetime.now().strftime("%M_%S_%MS")
        pickle.dump(self.reward, open(f"../rewards/{save_reward_string}-15-fast.pkl", "wb"))

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):
        if self.stop:
            return False
        while self.pause:
            self.myWorld.step_reward()
            time.sleep(0.1)

        if self.load:
            self.reward = self.load_rewards()
            # self.reward = self.load_rewards_without_outliers()
        else:
            self.calc_reward()
        # self.plot_rewards()
        # for x in np.nditer(self.reward, op_flags=['readwrite']):
        #     if abs(x) < 0:
        #         x[...] = 0
        self.save_reward_as_txt()

        for i in range(steps):
            print(i)
            self.improve_value_and_policy()
            self.ex.update_ui_step(self.steps)

        self.save_value_as_txt()
        self.ex.update_ui_finished()
        self.print_value_table()
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
        start = time.time()
        while not self.stop:
            while self.pause and not self.stop:
                time.sleep(0.1)
            a = self.get_greedy_action(self.state)
            state = self.myRobot.get_state().copy()
            successor_state = self.myRobot.apply_action(self.myRobot.action_to_diff[a])
            self.total_reward += self.myWorld.step_reward()
            self.steps += self.myWorld.get_steps_done()
            if self.total_reward >= 30:
                break
            # self.execute_reward[tuple(state.flatten()) + (a,)] = reward
            # print(rew)
            self.state = successor_state
        end = time.time()
        if self.load:
            self.myWorld.draw_steps()
            self.myWorld.draw_angles()
            self.myWorld.draw_states()
        print(str(self.steps) + " steps")
        print(str(self.total_reward) + " meter")
        print(str(round(self.total_reward / (end - start), 2)) + "m/s")
        self.save_reward_as_txt(execute=True)

    def plot_rewards(self):
        plot = sns.kdeplot(self.reward.flatten(), bw=0.2, multiple="stack")
        plt.xlabel("Reward")
        plt.ylabel("Häufigkeitsdichte")
        plt.savefig("rewards.png")
        plt.show()

    def load_rewards(self):
        reward_mean = np.zeros(self.reward.shape)
        std_devs = []
        files = [f for f in listdir("../rewards") if isfile(join("../rewards", f))]
        for file in files:
            opentxt = f"../rewards/{file}"
            savetxt = f"../single_reward_as_txt/{file}.txt"
            single_reward = pickle.load(open(opentxt, "rb"))
            self.save_reward_as_txt(single_reward=single_reward, txt=savetxt)
            reward_mean += single_reward
            std_devs.append(single_reward)
        self.std_dev = np.array(np.std(std_devs, axis=0))
        self.print_std_dev()
        return reward_mean / len(files)

    def print_std_dev(self):
        std_devs = self.std_dev.flatten()
        std_devs_without_zeros = std_devs[std_devs != 0]
        std_dev = np.std(std_devs_without_zeros)
        std_mean = np.mean(std_devs_without_zeros)
        print("Erwartungswert der Standardabweichungen: " + str(std_mean))
        print("Standardabweichung davon :" + str(std_dev))

    def load_rewards_without_outliers(self):
        vfunc = np.vectorize(self.my_func)
        std_devs = []
        rewards = []
        files = [f for f in listdir("../rewards") if isfile(join("../rewards", f))]
        for file in files:
            opentxt = f"../rewards/{file}"
            savetxt = f"../single_reward_as_txt/{file}.txt"
            single_reward = pickle.load(open(opentxt, "rb"))
            self.save_reward_as_txt(single_reward=single_reward, txt=savetxt)
            rewards.append(single_reward)
            std_devs.append(single_reward)
        self.std_dev = np.array(np.std(std_devs, axis=0))
        self.print_std_dev()
        return vfunc(*rewards)

    def my_func(self, *rewards):
        return stats.trim_mean([*rewards], 0.1)

    def save_value_as_txt(self):
        value_size = len(self.value.shape)
        with open("value_as_text.txt", "w") as text_file:
            for idx, value in np.ndenumerate(self.value):
                text_file.write("Zustand %s -> Value %s \n" % (
                    idx[0:value_size], value))

    def save_reward_as_txt(self, execute=False, single_reward=None, txt="reward_as_text.txt"):
        if single_reward is None:
            if self.std_dev is not None:
                reward = self.reward
                reward_size = len(reward.shape)
                with open(txt, "w") as text_file:
                    for idx, value in np.ndenumerate(reward):
                        text_file.write("Zustand %s : Aktion %s -> Reward %s | Std: %s \n" % (
                            idx[0:reward_size - 1], idx[reward_size - 1], value, self.std_dev[idx]))
            else:
                if execute:
                    reward = self.execute_reward
                    txt = "exe_reward_as_text.txt"
                else:
                    reward = self.reward
                reward_size = len(reward.shape)
                with open(txt, "w") as text_file:
                    for idx, value in np.ndenumerate(reward):
                        if value == 0.0:
                            continue
                        text_file.write("Zustand %s : Aktion %s -> Reward %s \n" % (
                            idx[0:reward_size - 1], idx[reward_size - 1], value))
        else:
            reward = single_reward
            reward_size = len(reward.shape)
            with open(txt, "w") as text_file:
                for idx, value in np.ndenumerate(reward):
                    text_file.write("Zustand %s : Aktion %s -> Reward %s \n" % (
                        idx[0:reward_size - 1], idx[reward_size - 1], value))

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
        :return: value of the action with the highest reward and the action itself as
        (value of action, action as number between 0 and 3)
        """
        max_a = None
        argmax_a = None
        for a in self.myRobot.get_possible_actions(state):
            long_term_reward = self.reward[tuple(state.flatten()) + (a,)] + self.GAMMA * self.value[
                tuple(self.myRobot.transition(state, a).flatten())]
            if max_a is None or long_term_reward > max_a:
                max_a = long_term_reward
                argmax_a = a
        return max_a, argmax_a

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
