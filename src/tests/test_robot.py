import logging
import numpy as np
import unittest
from robot import RobotDiscrete
from robot.Robot import Mode


class TestRobot(unittest.TestCase):
    """
    Unit Tests for the robot class
    """

    def setUp(self) -> None:
        self.robot = RobotDiscrete.RobotDiscrete()
        self.robot.init(Mode.ONE_ACTION, 1, 2, [[-20, 40], [40, 140]], [5, 5])

    def test_init_state_lookuptable_joints(self):
        self.assertEqual(len(self.robot.state_lookuptable_joints), 2)

    def test_init_action_to_diff(self):
        self.assertEqual(self.robot.action_to_diff, {0: ((1, 0),), 1: ((-1, 0),), 2: ((0, 1),), 3: ((0, -1),)})

    def test_init_state(self):
        self.assertTrue(np.array_equal(self.robot.state, np.array([[2, 2]])))

    def test_apply_action(self):
        next_state = self.robot.apply_action(((1, 0),))
        expected_next_state = np.array([[3, 2]])
        self.assertTrue(np.array_equal(next_state, expected_next_state))
        self.assertTrue(np.array_equal(self.robot.state, expected_next_state))

    def test_get_all_states(self):
        all_states = self.robot.get_all_states()
        expected_all_states = np.array([[[0, 0]], [[0, 1]], [[0, 2]], [[0, 3]], [[0, 4]],
                                        [[1, 0]], [[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]],
                                        [[2, 0]], [[2, 1]], [[2, 2]], [[2, 3]], [[2, 4]],
                                        [[3, 0]], [[3, 1]], [[3, 2]], [[3, 3]], [[3, 4]],
                                        [[4, 0]], [[4, 1]], [[4, 2]], [[4, 3]], [[4, 4]]])
        self.assertTrue(np.array_equal(all_states, expected_all_states))

    def test_get_possible_states_middle(self):
        possible_actions = self.robot.get_possible_actions(((2, 2),))
        expected_possible_actions = [0, 1, 2, 3]
        self.assertEqual(possible_actions, expected_possible_actions)

    def test_get_possible_states_edge(self):
        possible_actions = self.robot.get_possible_actions(((0, 2),))
        expected_possible_actions = [0, 2, 3]
        self.assertEqual(possible_actions, expected_possible_actions)

    def test_get_possible_states_corner(self):
        possible_actions = self.robot.get_possible_actions(((4, 0),))
        expected_possible_actions = [1, 2]
        self.assertEqual(possible_actions, expected_possible_actions)

    def test_transition(self):
        state = np.array(((2, 2),))
        state_transition = self.robot.transition(state, 1)
        expected_state_transition = np.array([[1, 2]])
        self.assertTrue(np.array_equal(state_transition, expected_state_transition))

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
