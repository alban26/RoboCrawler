import logging

from Box2D import *
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

import learning.DQN.DQNLearning as DQNLearning
import learning.QLearning as QLearning
import learning.WalkAround as WalkAround
import sim.PyQt5Draw as PyQt5Draw
import sim.World as World
import sim.settings as settings
from learning import ValueIteration
from sim import RobotSim

"""
Code is adapted from the official pybox2d example:
https://github.com/pybox2d/pybox2d/blob/master/examples/backends/pyqt4_framework.py
"""


class LearningThread(QtCore.QThread):
    """
    Thread for learning. This is done in an own thread to keep the GUI responsive.
    """

    draw_signal = QtCore.pyqtSignal()

    def __init__(self, learn_algo, ui):
        super().__init__()
        self.learning_algorithm = learn_algo
        self.ui = ui
        self.learning_steps = 200
        self.min_epsilon = 0.3
        self.max_epsilon = 1.0
        self.is_finished_learning = False
        self.hasPolicyLoaded = False

    def setPolicyAlreadyLoaded(self):
        """
        Sets flag if a policy has already been passed to the GUI
        :return:
        """
        logging.debug("Set variable in framework: hasPolicyLoaded = True\n")
        self.hasPolicyLoaded = True

    def set_learning_steps(self, steps):
        self.learning_steps = steps

    def set_epsilon_range(self, min_epsilon, max_epsilon):
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def set_improve_every_steps(self, improve_every_steps):
        self.improve_every_steps = improve_every_steps

    def set_is_finished(self, val=False):
        self.is_finished_learning = val

    # def set_tricks(self, tricks):
    #     self.tricks = tricks

    def set_invert_learning(self, invert_learning):
        self.invert_learning = invert_learning

    def set_learning_algorithm(self, learning_algorithm):
        self.learning_algorithm = learning_algorithm

    def run(self):
        """
        runs the specified learning algorithm
        :return:
        """
        if self.hasPolicyLoaded:
            logging.debug("Learning mode with loaded policy: \n")
            if self.learning_steps == 0:
                self.learning_algorithm.execute()
                self.is_finished_learning = False
                logging.debug("Finished executing")
            else:
                if not self.is_finished_learning:
                    self.is_finished_learning = self.learning_algorithm.learn(self.learning_steps, self.min_epsilon,
                                                                              self.max_epsilon,
                                                                              self.improve_every_steps,
                                                                              self.invert_learning, self.ui)
                    if self.is_finished_learning:
                        logging.debug("Finished learning")
                        self.ui.execute_button.setEnabled(True)
                else:
                    self.learning_algorithm.execute()
                    self.is_finished_learning = False
                    logging.debug("Finished executing")
        else:  # run normally
            logging.debug("Learning mode without pre-loaded policy:")
            logging.debug("is_finished_learning: {}\n".format(self.is_finished_learning))
            if not self.is_finished_learning:
                self.is_finished_learning = self.learning_algorithm.learn(self.learning_steps, self.min_epsilon,
                                                                          self.max_epsilon, self.improve_every_steps,
                                                                          self.invert_learning, self.ui)
                if self.is_finished_learning:
                    logging.debug("Finished learning")
                    self.ui.execute_button.setEnabled(True)
            else:
                logging.debug("Executing policy now\n")
                self.learning_algorithm.execute()
                self.is_finished_learning = False
                logging.debug("Finished executing")

    def __del__(self):
        logging.debug("Deleted Learning Thread")


class Framework:
    """
    This class is instantiated within the gui class and contains all relevant elements for the running gui,
    such as the learning algorithm, the graphical simulation and the connection towards the real robot.
    """

    def __init__(self, window, robot, ui, ex):
        self.window = window
        self.renderer = PyQt5Draw.Pyqt5Draw(self)
        self.robot = robot
        self.settings = settings.Settings()
        self.ex = ex

        # Create Box2d world
        self.b2World = b2World(gravity=(0, -10), doSleep=True)
        self.b2World.DrawDebugData = lambda: self.renderer.ManualDraw()
        self.b2World.renderer = self.renderer
        # self.b2World.warmStarting = True
        # self.b2World.continuousPhysics = True
        # self.b2World.subStepping = True

        # Initial all
        self.robot_sim = RobotSim.RobotSim(self.robot, [1, 1], [(1.2, 0.1), (0.8, 0.1)], [1, 1], self.b2World)
        self.world = World.World(self.renderer, self.settings, self.robot, self.robot_sim, self.b2World, self.ex)
        self.learning_algorithm = ValueIteration.ValueIteration(self.world.robot, self.world, self.ex)
        self.set_pause(False)
        self.set_stop(False)
        self.learning_thread = LearningThread(self.learning_algorithm, ui)
        self.learning_thread.draw_signal.connect(self.world.draw)
        self.world.set_draw_signal(self.learning_thread.draw_signal)
        self.reset()

    def set_is_finished_learing(self, val=False):
        self.learning_thread.set_is_finished(val)

    def setPolicyAlreadyLoaded(self):
        self.learning_thread.setPolicyAlreadyLoaded()

    def set_learning_steps(self, steps):
        self.learning_thread.set_learning_steps(steps)

    def set_epsilon_range(self, min_epsilon, max_epsilon):
        self.learning_thread.set_epsilon_range(min_epsilon, max_epsilon)

    def set_improve_every_steps(self, improve_every_steps):
        self.learning_thread.set_improve_every_steps(improve_every_steps)

    # def set_tricks(self, tricks):
    #     self.learning_thread.set_tricks(tricks)

    def set_invert_learning(self, invert_learning):
        self.learning_thread.set_invert_learning(invert_learning)

    def set_learning_algorithm(self, learning_algorithm):
        if learning_algorithm == 0:
            self.learning_algorithm = ValueIteration.ValueIteration(self.world.robot, self.world, self.ex)
        elif learning_algorithm == 1:
            self.learning_algorithm = QLearning.QLearning(self.world.robot, self.world, self.ex)
        elif learning_algorithm == 2:
            self.learning_algorithm = DQNLearning.DQNLearning(self.world.robot, self.world, self.ex)
        elif learning_algorithm == 3:
            self.learning_algorithm = WalkAround.WalkAround(self.world.robot, self.world, self.ex)

        self.learning_thread.set_learning_algorithm(self.learning_algorithm)
        self.reset()

    def reset_learn_algo(self):
        self.learning_algorithm.reset()

    def reset(self):
        """
        reset the view to default settings
        :return:
        """
        self.renderer.reset()
        self.robot_sim.reset()
        self.world.reset()
        self.learning_algorithm.reset()
        self.settings = settings.Settings()
        self.screenSize = b2Vec2(0, 0)
        self.setCenter((0, 10.0 * 20.0))
        self.setZoom(45.0)

    def setCenter(self, value):
        """
        Updates the view offset based on the center of the screen.
        Tells the debug draw to update its values as well.
        :return:
        """
        self.viewCenter = b2Vec2(*value)
        self.viewOffset = self.viewCenter - self.screenSize / 2
        self.window.graphicsView.centerOn(*self.viewCenter)

    def setZoom(self, zoom):
        """
        :param zoom: amount of zoom
        :return:
        """
        self.viewZoom = zoom
        self.window.graphicsView.resetTransform()
        self.window.graphicsView.scale(self.viewZoom, -self.viewZoom)
        self.window.graphicsView.centerOn(*self.viewCenter)

    def start(self):
        """
        Lets the robot simulation learn.
        :return:
        """
        if not self.learning_thread.isRunning():
            logging.debug("thread is not running")
            self.learning_thread.start()
        else:
            logging.debug("thread is running")

    def save_policy(self, file_path, param_vector):
        self.learning_algorithm.save_policy(file_path, param_vector)

    def load_policy(self, file_path):
        self.learning_algorithm.load_policy(file_path)

    def get_pause(self):
        """
        :return: boolean that tells if the learning is paused or not.
        """
        return self.learning_algorithm.get_pause()

    def set_pause(self, val=True):
        """
        Controls whether the learning is paused or not.
        :param val: True by default
        :return:
        """
        self.learning_algorithm.set_pause(val)

    def set_stop(self, val=True):
        """
        Stops the algorithm from learning any further.
        :param val: True by default.
        :return:
        """
        self.learning_algorithm.set_stop(val)

    def _Keyboard_Event(self, key, down=True):
        """
        Internal keyboard event, don't override this.

        Checks for the initial keydown of the basic testbed keys. Passes the unused
        ones onto the test via the Keyboard() function.

        :param key: pressed key
        :param down:
        :return:
        """
        if down:
            if key == Qt.Key_Z:  # Zoom in
                self.setZoom(min(1.10 * self.viewZoom, 50.0))
            elif key == Qt.Key_X:  # Zoom out
                self.setZoom(max(0.9 * self.viewZoom, 0.02))
            else:  # Inform the test of the key press
                self.Keyboard(key)
        else:
            self.KeyboardUp(key)

    def Keyboard(self, key):
        """
        Callback indicating 'key' has been pressed down.
        The keys are mapped after pygame's style.

         from framework import Keys
         if key == Keys.K_z:
             ...


         :param key: pressed key
         :return:
        """
        pass

    def KeyboardUp(self, key):
        """
        Callback indicating 'key' has been released.
        See Keyboard() for key information

        :param key: pressed key
        :return:
        """
        pass
