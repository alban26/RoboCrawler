for step in range(self.STEPS_PER_EPISODE):
    action = self.select_action(state=state, epsilon=epsilon)
    next_state = self.myRobot.apply_action(self.myRobot.action_to_diff[action.item()])
    reward = self.myWorld.step_reward()
    self.memory.append(Experience(state, action, reward, next_state))