import logging
import pickle
import sys
from tkinter import filedialog

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from gui.gui import Ui_MainWindow
from robot.Robot import Mode
from robot.RobotDiscrete import RobotDiscrete
from sim.framework import Framework
matplotlib.rcParams.update({'font.size': 12})


class Ex(QMainWindow):
    """
    Class that contains and handles GUI elements.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #  Thread
        # self.myThread = threading.Thread(target=self.createThreadForLearning)
        self.serialEnabled = False

        self.ui.start_learning_button.clicked.connect(self.clickedOnStartLearning)
        self.ui.stop_robot_button.clicked.connect(self.clickedOnStopRobot)
        self.ui.pause_learning_button.clicked.connect(self.clickedOnPauseLearning)
        self.ui.execute_button.clicked.connect(self.clickedOnExecute)
        self.ui.save_policy_button.clicked.connect(self.clickedOnSave)
        self.ui.load_policy_button.clicked.connect(self.clickedOnLoad)

        self.ui.param_min_deg_j1_txt.setText("-20")
        self.ui.param_max_deg_j1_txt.setText("30")
        self.ui.param_deg_steps_j1_txt.setText("6")
        self.ui.param_min_deg_j2_txt.setText("30")
        self.ui.param_max_deg_j2_txt.setText("130")
        self.ui.param_deg_steps_j2_txt.setText("10")
        self.ui.param_learning_steps_txt.setText("100")
        self.ui.epsilon_min_txt.setText("0.3")
        self.ui.epsilon_max_txt.setText("1.0")
        self.ui.param_improve_every_steps_txt.setText("1")
        self.ui.speed_slider.setValue(0)
        self.ui.servo_speed_slider.setValue(0)

        self.is_executing = False

        self.graphicsView = QGraphicsView()
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setMinimumHeight(300)
        self.graphicsView.setMinimumWidth(300)

        self.ui.verticalLayout_simu.addWidget(self.graphicsView)

        #  Parameters

        self.robot = RobotDiscrete()
        self.updateRobotAttributesFromGUI()

        self.fw = Framework(self, self.robot, self.ui, self)

        self.ui.pause_learning_button.setEnabled(False)
        self.ui.stop_robot_button.setEnabled(False)
        self.ui.execute_button.setEnabled(False)

        self.hasPolicyLoaded = False  # variable is set after loading a policy to prevent overwriting it after starting

        self.fig = plt.figure()

        self.plotWidget = FigureCanvas(self.fig)
        lay = QtWidgets.QVBoxLayout(self.ui.value_graph_widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)

    def resizeEvent(self, event):
        """
        Resizes value graph when the window size changes
        :param event:
        :return:
        """
        self.draw_value_sum()
        QMainWindow.resizeEvent(self, event)

    def update_ui_step(self, steps, epsilon=1):
        """
        Update step label in GUI
        :param steps: amount of learning iterations
        :param epsilon: exploration factor
        :return:
        """
        if self.ui.cbDrawSumValueTable.isChecked():
            self.draw_value_sum()
        self.ui.label_steps.setText("Step: " + str(steps) + "       Epsilon: " + str(round(epsilon, 4)))

    def update_ui_step_dqn(self, steps, epsilon=1):
        """
        Update step label in GUI in DQN implementation
        :param steps: amount of learning iterations
        :param epsilon: exploration factor
        :return:
        """
        self.draw_mean_reward()
        self.ui.label_steps.setText("Step: " + str(steps) + "       Epsilon: " + str(round(epsilon, 4)))

    def update_ui_finished(self):
        """
        UI update as soon as the robot finished learning
        :return:
        """
        self.ui.label_steps.setText("Finished learning.")
        self.ui.pause_learning_button.setEnabled(False)
        self.ui.save_policy_button.setEnabled(True)

    def draw_value_sum(self):
        """
        Draw Value Graph
        :return:
        """
        plt.plot(self.fw.learning_algorithm.sum_table())
        plt.xlabel("Iteration")
        plt.ylabel("Wertsumme")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.clf()

    def draw_mean_reward(self):
        plt.plot(self.fw.learning_algorithm.mean_reward())
        plt.xlabel("Episoden")
        plt.ylabel("Reward Mittelwert")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.clf()

    def updateRobotAttributesFromGUI(self):
        """
        retrieve the robot attributes from the MainWindows TextFields and assign those to the robot's attributes
        :return:
        """

        self.joint_num = 2  # always 2 currently, there are no robot arms with more or less joints than two

        # GUI Radio Button
        if self.ui.one_arm_rb.isChecked():
            self.robo_mode = Mode.ONE_ACTION
        if self.ui.two_arm_all_action_rb.isChecked():
            self.robo_mode = Mode.ALL_POSSIBLE

        # set arm numbers depending on the robot's mode
        if self.robo_mode == Mode.ONE_ACTION:
            self.arm_num = 1
        else:
            self.arm_num = 2

        # GUI Parameters 1 to 6
        try:
            self.joint1_minimum_degree = int(self.ui.param_min_deg_j1_txt.text())
            self.joint1_maximum_degree = int(self.ui.param_max_deg_j1_txt.text())
            self.joint1_steps = int(self.ui.param_deg_steps_j1_txt.text())
            self.joint2_minimum_degree = int(self.ui.param_min_deg_j2_txt.text())
            self.joint2_maximum_degree = int(self.ui.param_max_deg_j2_txt.text())
            self.joint2_steps = int(self.ui.param_deg_steps_j2_txt.text())
        except:
            logging.debug("Either one of the joint parameters was not an int value, setting all to default")
            self.joint1_minimum_degree = -10
            self.ui.param_min_deg_j1_txt.setText("-10")
            self.joint1_maximum_degree = 40
            self.ui.param_max_deg_j1_txt.setText("40")
            self.joint1_steps = 3
            self.ui.param_deg_steps_j1_txt.setText("3")
            self.joint2_minimum_degree = 40
            self.ui.param_min_deg_j2_txt.setText("40")
            self.joint2_maximum_degree = 140
            self.ui.param_max_deg_j2_txt.setText("140")
            self.joint2_steps = 5
            self.ui.param_deg_steps_j2_txt.setText("5")

        self.robot.init(self.robo_mode, self.arm_num, self.joint_num,
                        [[self.joint1_minimum_degree, self.joint1_maximum_degree],
                         [self.joint2_minimum_degree, self.joint2_maximum_degree]],
                        [self.joint1_steps, self.joint2_steps])

        try:
            self.learning_steps = int(self.ui.param_learning_steps_txt.text())
        except:
            logging.debug("Int value was not entered for learning steps, using 250 instead")
            self.leaarning_steps = 250
            self.ui.param_learning_steps_txt.setText("250")

        try:
            self.min_epsilon = float(self.ui.epsilon_min_txt.text())
        except:
            logging.debug("Float value was not entered for min. epsilon, using 0.3 instead")
            self.max_epsilon = 0.3
            self.ui.epsilon_min_txt.setText("0.3")
        # check if self.min_epsilon is out of allowed range
        # 0.1 is the minimum for epsilon, because e = 0.0 would mean pure exploitation which is not intended while exploring
        if self.min_epsilon > 1.0 or self.min_epsilon < 0.1:
            self.ui.epsilon_min_txt.setText("0.3")
            logging.debug("min. epsilon value not valid: " + str(
                self.min_epsilon) + "\nPlease choose value above/equal 0.1 and smaller/equal 1.0, using 0.3 instead")
            self.min_epsilon = 0.3

        try:
            self.max_epsilon = float(self.ui.epsilon_max_txt.text())
        except:
            logging.debug("Float value was not entered for max. epsilon, using 1.0 instead")
            self.max_epsilon = 1.0
            self.ui.epsilon_max_txt.setText("1.0")
        # check if self.max_epsilon is out of allowed range
        if self.max_epsilon > 1.0 or self.max_epsilon < 0.1:
            self.ui.epsilon_max_txt.setText("1.0")
            logging.debug("max. epsilon value not valid: " + str(
                self.max_epsilon) + "\nPlease choose value above/equal 0.1 and smaller/equal 1.0, using 1.0 instead")
            self.max_epsilon = 1.0

        # swap min and max epsilon if max_value is smaller than min_value
        if self.max_epsilon < self.min_epsilon:
            tmp = self.max_epsilon
            self.max_epsilon = self.min_epsilon
            self.min_epsilon = tmp

        try:
            self.improve_every_steps = int(self.ui.param_improve_every_steps_txt.text())
        except:
            logging.debug("Int value was not entered for improve every steps, using 1 instead")
            self.improve_every_steps = 1
            self.ui.param_learning_steps_txt.setText("1")

        #        self.tricks = self.ui.cbTricks.isChecked()
        self.invert_learning = self.ui.cbInvertLearning.isChecked()

        self.learning_algorithm = self.ui.choose_learning_algorithm.currentIndex()

    def updateGUIFromRobotAttributes(self):

        if self.robo_mode == Mode.ONE_ACTION:
            self.ui.one_arm_rb.setChecked(True)
        if self.robo_mode == Mode.ALL_POSSIBLE:
            self.ui.two_arm_all_action_rb.setChecked(True)

        self.ui.param_min_deg_j1_txt.setText(str(self.joint1_minimum_degree))
        self.ui.param_max_deg_j1_txt.setText(str(self.joint1_maximum_degree))
        self.ui.param_deg_steps_j1_txt.setText(str(self.joint1_steps))
        self.ui.param_min_deg_j2_txt.setText(str(self.joint2_minimum_degree))
        self.ui.param_max_deg_j2_txt.setText(str(self.joint2_maximum_degree))
        self.ui.param_deg_steps_j2_txt.setText(str(self.joint2_steps))
        self.ui.param_learning_steps_txt.setText(str(self.learning_steps))
        self.ui.epsilon_min_txt.setText(str(self.min_epsilon))
        self.ui.epsilon_max_txt.setText(str(self.max_epsilon))
        self.ui.param_improve_every_steps_txt.setText(str(self.improve_every_steps))
        self.ui.cbInvertLearning.setChecked(self.invert_learning)
        self.ui.choose_learning_algorithm.setCurrentIndex(self.learning_algorithm)

    def save_param_vector(self):

        # GUI Radio Button
        if self.ui.one_arm_rb.isChecked():
            robot_mode = Mode.ONE_ACTION
        if self.ui.two_arm_all_action_rb.isChecked():
            robot_mode = Mode.ALL_POSSIBLE

        joint1_min = int(self.ui.param_min_deg_j1_txt.text())
        joint1_max = int(self.ui.param_max_deg_j1_txt.text())
        joint1_steps = int(self.ui.param_deg_steps_j1_txt.text())
        joint2_min = int(self.ui.param_min_deg_j2_txt.text())
        joint2_max = int(self.ui.param_max_deg_j2_txt.text())
        joint2_steps = int(self.ui.param_deg_steps_j2_txt.text())
        learning_steps = int(self.ui.param_learning_steps_txt.text())
        min_epsilon = float(self.ui.epsilon_min_txt.text())
        max_epsilon = float(self.ui.epsilon_max_txt.text())
        improve_every_steps = int(self.ui.param_improve_every_steps_txt.text())
        invert_learning = self.ui.cbInvertLearning.isChecked()
        learning_algorithm = self.ui.choose_learning_algorithm.currentIndex()

        # learning steps is 0 by default when saving, so that the user has to configure it himself if he wants to resume learning
        return np.array(
            [robot_mode, joint1_min, joint1_max, joint1_steps, joint2_min, joint2_max, joint2_steps, learning_steps,
             min_epsilon, max_epsilon, improve_every_steps, invert_learning, learning_algorithm])

    def set_gui_params_from_vec(self, gui_params):
        self.robo_mode, self.joint1_minimum_degree, self.joint1_maximum_degree, self.joint1_steps, \
        self.joint2_minimum_degree, self.joint2_maximum_degree, self.joint2_steps, self.learning_steps, \
        self.min_epsilon, self.max_epsilon, self.improve_every_steps, self.invert_learning, \
        self.learning_algorithm = gui_params

    def clickedOnSave(self):
        file_path = filedialog.asksaveasfilename(
            filetypes=(("Text File", "*.p"), ("All Files", "*.*")),
            title="Choose a file.")
        if file_path:
            param_vector = self.save_param_vector()
            self.fw.save_policy(file_path, param_vector)
        else:
            logging.debug("No valid file location. Please choose a file path.")

    def clickedOnLoad(self):
        file_path = filedialog.askopenfilename(
            filetypes=(("Text File", "*.p"), ("All Files", "*.*")),
            title="Choose a file.")
        if file_path:
            gui_params = pickle.load(open(file_path, "rb"))[1]
            self.set_gui_params_from_vec(gui_params)
            self.updateGUIFromRobotAttributes()
            self.policy_filepath = file_path
            self.hasPolicyLoaded = True
            self.ui.execute_button.setEnabled(True)
        else:
            logging.debug("No file chosen. Please choose a file.")

    def clickedOnStopRobot(self):
        self.fw.set_stop()
        self.ui.start_learning_button.setEnabled(True)
        self.ui.pause_learning_button.setEnabled(False)
        self.ui.stop_robot_button.setEnabled(False)
        self.ui.execute_button.setEnabled(False)
        self.ui.save_policy_button.setEnabled(True)
        self.ui.load_policy_button.setEnabled(True)
        self.ui.execute_button.setText("Execute Policy")
        self.enableWhileNotRunning()
        self.is_executing = False

    def clickedOnPauseLearning(self):
        if self.fw.get_pause():
            self.ui.pause_learning_button.setText("Pause Learning")
            self.fw.set_pause(False)
            self.ui.stop_robot_button.setEnabled(True)
            self.ui.save_policy_button.setEnabled(False)
        else:
            self.fw.set_pause()
            self.ui.pause_learning_button.setText("Resume Learning")
            self.ui.stop_robot_button.setEnabled(False)
            self.ui.save_policy_button.setEnabled(True)

    def clickedOnStartLearning(self):
        # update robot parameters from gui inputs
        self.updateRobotAttributesFromGUI()

        self.fw.reset()

        self.disableWhileRunning()
        self.fw.set_pause(False)  # run the val_it
        self.fw.set_stop(False)
        self.fw.set_learning_steps(self.learning_steps)
        self.fw.set_epsilon_range(self.min_epsilon, self.max_epsilon)
        self.fw.set_improve_every_steps(self.improve_every_steps)
        self.fw.set_invert_learning(self.invert_learning)
        self.fw.set_learning_algorithm(self.learning_algorithm)

        if self.hasPolicyLoaded:
            logging.debug("Detected a pre-loaded policy!\n")
            self.fw.setPolicyAlreadyLoaded()
            self.fw.load_policy(self.policy_filepath)
            self.hasPolicyLoaded = False

        self.fw.set_is_finished_learing()
        self.fw.start()

        self.ui.pause_learning_button.setEnabled(True)
        self.ui.stop_robot_button.setEnabled(True)
        self.ui.save_policy_button.setEnabled(False)
        self.ui.load_policy_button.setEnabled(False)

    def clickedOnExecute(self):
        if self.hasPolicyLoaded:
            logging.debug("Executing with loaded policy...")

            # update robot params
            self.updateRobotAttributesFromGUI()
            self.fw.reset()

            self.fw.set_learning_steps(self.learning_steps)
            self.fw.set_epsilon_range(self.min_epsilon, self.max_epsilon)
            self.fw.set_improve_every_steps(self.improve_every_steps)
            self.fw.set_invert_learning(self.invert_learning)
            self.fw.set_learning_algorithm(self.learning_algorithm)

            self.fw.setPolicyAlreadyLoaded()
            self.fw.load_policy(self.policy_filepath)
            self.hasPolicyLoaded = False
            self.fw.set_is_finished_learing(True)
            self.fw.set_stop(False)

        if not self.is_executing:
            self.fw.start()
            self.is_executing = True
            self.fw.set_pause()
            self.draw_value_sum()
            self.ui.stop_robot_button.setEnabled(True)
        if self.fw.get_pause():
            self.ui.execute_button.setText("Pause Executing")
            self.fw.set_pause(False)
        else:
            self.fw.set_pause()
            self.ui.execute_button.setText("Resume Executing")
            self.ui.stop_robot_button.setEnabled(True)
        self.ui.start_learning_button.setEnabled(False)

    def setLearnedSteps(self):
        self.ui.label_steps.setText("Steps: " + "")

    def disableWhileRunning(self):
        self.ui.start_learning_button.setEnabled(False)
        self.ui.param_min_deg_j1_txt.setEnabled(False)
        self.ui.param_max_deg_j1_txt.setEnabled(False)
        self.ui.param_min_deg_j2_txt.setEnabled(False)
        self.ui.param_max_deg_j2_txt.setEnabled(False)
        self.ui.param_deg_steps_j1_txt.setEnabled(False)
        self.ui.param_deg_steps_j2_txt.setEnabled(False)
        self.ui.servo_speed_slider.setEnabled(False)
        self.ui.epsilon_max_txt.setEnabled(False)
        self.ui.epsilon_min_txt.setEnabled(False)
        self.ui.param_learning_steps_txt.setEnabled(False)
        self.ui.param_improve_every_steps_txt.setEnabled(False)
        self.ui.one_arm_rb.setEnabled(False)
        self.ui.two_arm_all_action_rb.setEnabled(False)
        self.ui.cbInvertLearning.setEnabled(False)
        self.ui.choose_learning_algorithm.setEnabled(False)

    def enableWhileNotRunning(self):
        self.ui.start_learning_button.setEnabled(True)
        self.ui.param_min_deg_j1_txt.setEnabled(True)
        self.ui.param_max_deg_j1_txt.setEnabled(True)
        self.ui.param_min_deg_j2_txt.setEnabled(True)
        self.ui.param_max_deg_j2_txt.setEnabled(True)
        self.ui.param_deg_steps_j1_txt.setEnabled(True)
        self.ui.param_deg_steps_j2_txt.setEnabled(True)
        self.ui.servo_speed_slider.setEnabled(True)
        self.ui.epsilon_max_txt.setEnabled(True)
        self.ui.epsilon_min_txt.setEnabled(True)
        self.ui.param_learning_steps_txt.setEnabled(True)
        self.ui.param_improve_every_steps_txt.setEnabled(True)
        self.ui.one_arm_rb.setEnabled(True)
        self.ui.two_arm_all_action_rb.setEnabled(True)
        self.ui.cbInvertLearning.setEnabled(True)
        self.ui.choose_learning_algorithm.setEnabled(True)


def start():
    app = QApplication(sys.argv)  # A new instance of QApplication
    form = Ex()  # We set the form to be our ExampleApp (design)
    form.show()  # Show the form
    app.exec_()
