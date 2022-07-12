import random

import numpy as np

from learning.LearningAlgorithm import LearningAlgorithm


class WalkAround(LearningAlgorithm):

    def get_policy(self):
        return self.value, self.reward

    def set_policy(self, policy):
        (self.value, self.reward) = policy

    def get_table(self):
        return self.value

    def execute(self):
        pass

    def __init__(self, myRobot, myWorld, ex):
        super().__init__(myRobot, myWorld, ex)
        self.reward = None
        self.policy = None
        self.value = None
        self.last_action_diff = None

    def reset(self):

        super().reset()

        self.value = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.policy = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num)
        self.reward = np.zeros(self.myRobot.joints_states_num * self.myRobot.arms_num + [self.myRobot.action_size()])
        self.state = self.myRobot.get_state()
        self.last_action_diff = np.zeros(self.state.shape)

    def calculate_reward(self, tricks):
        pass

    def learn(self, steps, min_epsilon, max_epsilon, improve_every_steps, invert_learning, ui):

        self.walk_tui()

        return True

    def walk_tui(self):
        while True:
            print("---------------------------------------------------------------------------------------")
            print("current state \narm 1 %s arm 2 %s\n" % (self.myRobot.state[0], self.myRobot.state[1]))

            joints = self.myRobot.joints_states_num
            print(
                "change state:\n > state {arm1-joint1 (0-%s)} {arm1-joint2 (0-%s)} {arm2-joint1 (0-%s)} {arm2-joint2 ("
                "0-%s)} "
                % (joints[0] - 1, joints[1] - 1, joints[0] - 1, joints[1] - 1))
            print(self.myRobot.statenr_to_state, "\n")

            print("Following states are possible - execute action by :\n > action {action number}")
            for n in self.myRobot.get_possible_actions(self.myRobot.state):
                print(n, "->", self.myRobot.action_to_diff[n], end=" ")
            print("\n")
            print("Or just walk some steps by\n > walk {number of steps} \n")
            print("To compare transitions \n > compare t \n")

            user_input = input()
            self.process_input(user_input)

    def process_input(self, user_input):
        inputs = user_input.split()
        if len(inputs) >= 2:
            if inputs[0] == "walk":
                self.walk(int(inputs[1]))
            if inputs[0] == "walkreverse":
                self.walk_reverse(int(inputs[1]))
            if inputs[0] == "state":
                self.change_state(inputs)
            if inputs[0] == "action":
                self.take_action(inputs)
            if inputs[0] == "compare":
                if inputs[1] == "t":
                    self.compare_transitions()
                else:
                    print("Please check your input")
        else:
            print("Please check your input")

    def take_action(self, action):
        degrees = self.myRobot.get_arm_states_degree()
        action_diff = ((int(action[1]), int(action[2])), (int(action[3]), int(action[4])))
        self.myRobot.apply_action(action_diff)
        degrees_after = self.myRobot.get_arm_states_degree()
        degrees_delta = np.subtract(degrees, degrees_after)
        print("Winkel geändert um: " + str(degrees_delta))
        reward = self.myWorld.step_reward()
        # self.myWorld.draw_steps()
        print("reward for done action:", reward)

    def change_state(self, new_state):
        self.myRobot.state = np.array([[int(new_state[1]), int(new_state[2])], [int(new_state[3]), int(new_state[4])]])
        self.myWorld.step_reward()
        print(self.myRobot.get_arm_states_degree())

    def walk(self, steps):
        steps = int(steps / 2)
        actionsLeft = []
        for i in range(steps):
            for i in range(2):
                actionsLeft.append((-1, 0))
            for i in range(8):
                actionsLeft.append((0, -1))
            for i in range(2):
                actionsLeft.append((1, 0))
            for i in range(8):
                actionsLeft.append((0, 1))
        actions = []
        td = 10  # time shift between left and right
        for i in range(td):
            actions.append((actionsLeft[i], (0, 0)))
        for i in range(td, len(actionsLeft)):
            actions.append((actionsLeft[i], actionsLeft[i - td]))

        for a in actions:
            self.myRobot.apply_action(a)
            rew = self.myWorld.step_reward()
            print(rew)
        print("Steps done: " + str(self.myWorld.get_steps_done()) + "\n")
        start_state = self.myRobot.joints_states_num
        self.myRobot.state = np.array(
            [[int(start_state[0] / 2), int(start_state[1] / 2)], [int(start_state[0] / 2), int(start_state[1] / 2)]])
        # self.myWorld.step_reward()
        # self.myWorld.draw_steps()

    def walk_reverse(self, steps):
        steps = int(steps / 2)
        actionsLeft = []
        for i in range(steps):
            for i in range(2):
                actionsLeft.append((0, 1))
            for i in range(8):
                actionsLeft.append((1, 0))
            for i in range(2):
                actionsLeft.append((0, -1))
            for i in range(8):
                actionsLeft.append((-1, 0))
        actions = []
        td = 10  # time shift between left and right
        for i in range(td):
            actions.append((actionsLeft[i], (0, 0)))
        for i in range(td, len(actionsLeft)):
            actions.append((actionsLeft[i], actionsLeft[i - td]))

        for a in actions:
            self.myRobot.apply_action(a)
            rew = self.myWorld.step_reward()
            print(rew)
        print("Steps done: " + str(self.myWorld.get_steps_done()) + "\n")
        start_state = self.myRobot.joints_states_num
        self.myRobot.state = np.array(
            [[int(start_state[0] / 2), int(start_state[1] / 2)], [int(start_state[0] / 2), int(start_state[1] / 2)]])
        self.myWorld.step_reward()
        # self.myWorld.draw_steps()

    def get_all_possible_state_action(self):
        state_action_pairs = []
        for state in self.myRobot.get_all_states():
            for action in self.myRobot.get_possible_actions(state):
                state_action_pairs.append((state, action))
        random.shuffle(state_action_pairs)
        return state_action_pairs

    def compare_transitions(self):
        state_action_nextstate_1 = []
        state_action_nextstate_2 = []
        all_possible_state_action_pairs = self.myRobot.get_all_possible_state_action()
        for i in range(2):
            n = 0
            N = len(all_possible_state_action_pairs)
            for state_action in self.get_all_possible_state_action():
                self.myRobot.state = state_action[0].copy()
                self.state = state_action[0].copy()
                self.myWorld.reset_sim()
                action_diff = self.myRobot.action_to_diff[state_action[1]]

                next_state = self.myRobot.apply_action(action_diff)

                reward = self.myWorld.step_reward()

                if i == 0:
                    state_action_nextstate_1.append(
                        (self.state[0][0], self.state[0][1],
                         self.state[1][0], self.state[1][1],
                         state_action[1],
                         self.myRobot.state[0][0], self.myRobot.state[0][1],
                         self.myRobot.state[1][0], self.myRobot.state[0][1],
                         reward)
                    )
                else:
                    state_action_nextstate_2.append(
                        (self.state[0][0], self.state[0][1],
                         self.state[1][0], self.state[1][1],
                         state_action[1],
                         self.myRobot.state[0][0], self.myRobot.state[0][1],
                         self.myRobot.state[1][0], self.myRobot.state[0][1],
                         reward)
                    )
                n += 1
                if n % 1000 == 0:
                    print('\rIteration {}/{}'.format(n, N), end="")
                if n == 20000:
                    break

        # Compares the state, action transitions
        state_action_nextstate_1.sort(key=lambda tup: tup[1])
        state_action_nextstate_2.sort(key=lambda tup: tup[1])

        for l1 in state_action_nextstate_1:
            for l2 in state_action_nextstate_2:
                if l1[0:5] == l2[0:5]:
                    if l1[5:9] != l2[5:9]:
                        print("Alarm" + str(l1) + str(l2))
                    print(l1, l2)

        print(set(state_action_nextstate_1) ^ set(state_action_nextstate_2))

    def improve_value_and_policy(self):
        pass

    def get_greedy_action(self, state):
        pass

    def max_long_term_reward(self, state):
        pass

    def print_value_table(self):
        pass
