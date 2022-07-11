import logging
import time
from Box2D import *

from robot.RobotDiscrete import RobotDiscrete
from timeit import default_timer as timer


class World:
    """
    The world defined in Box2D physic.
    """

    def __init__(self, renderer, settings, robot, robot_sim, b2World, ex):
        """
        :param renderer: PyQt5Draw object to render simulation
        :param settings: settings
        :param robot: Robot (RobotContinuous or RobotDiscrete)
        :param robot_sim: RobotSim
        :param b2World: b2World from Box2D
        :param comm: Crawler_Communicator
        """
        self.steps_done = 0
        self.renderer = renderer
        self.settings = settings
        self.reward_cap = 0.2  # cap reward under 1 to 0
        self.robot = robot
        self.robot_sim = robot_sim
        self.b2World = b2World
        self.ex = ex
        self.steps = 0
        self.use_real_robot = False
        self.velocities = []

        self.init()

    def init(self):
        """
        Init World (after reset)
        :return:
        """
        self.add_floor()

    def __del__(self):
        self.b2World = None

    def add_floor(self):
        """
        Adds a floor to the Box2D world.
        :return:
        """
        groundBodyDef = b2BodyDef()
        groundBodyDef.position = (0, -0.4)
        groundBody = self.b2World.CreateBody(groundBodyDef)
        groundBox = b2PolygonShape(box=(500000, 0.4))
        groundBoxFixture = b2FixtureDef(shape=groundBox, categoryBits=0x0004, maskBits=0x0002)
        groundBody.CreateFixture(groundBoxFixture)

    def draw(self):
        """
        Draws the world.
        :return:
        """
        self.renderer.StartDraw()
        self.b2World.DrawDebugData()
        self.renderer.EndDraw()

    def set_draw_signal(self, draw_signal):
        """
        Sets the draw signal, because the rendering in PyQt has to be done in a thread with render context.
        :param draw_signal:
        :return:
        """
        self.draw_signal = draw_signal

    def get_params(self):
        return self.ex.ui.cbDrawRobot.isChecked(), float(self.ex.ui.speed_slider.value()) / 10000

    def reset_sim(self):
        draw_robot, speed = self.get_params()
        while True:
            if isinstance(self.robot, RobotDiscrete):
                if self.robot_sim.update():
                    break
            if self.settings.hz > 0.0:
                time_step = 1.0 / self.settings.hz
            else:
                time_step = 0.0
            self.b2World.Step(time_step, self.settings.velocityIterations, self.settings.positionIterations)

            self.b2World.ClearForces()

            if draw_robot:
                self.draw_signal.emit()  # calls self.draw() in other thread with render context (Qt)

        self.robot_sim.reset_velocity()

    def x_pos(self):
        return self.robot_sim.get_body_pos()[0]

    def step_reward(self):
        """
        Steps and renders until goal state is reached.
        :return: reward
        """
        draw_robot, speed = self.get_params()

        robot_start_pos = self.robot_sim.get_body_pos()
        steps = 0
        counter = 0
        while True:  # Simulate until next/goal state of robot is reached
            if isinstance(self.robot, RobotDiscrete):
                if self.robot_sim.update():
                    break
            if counter % 100 == 0:
                self.robot_sim.step_counter()
                # self.robot_sim.collect_steps()

            counter += 1
            if self.settings.hz > 0.0:
                time_step = 1.0 / self.settings.hz
            else:
                time_step = 0.0
            self.b2World.Step(time_step, self.settings.velocityIterations, self.settings.positionIterations)

            self.b2World.ClearForces()

            if draw_robot:
                self.draw_signal.emit()  # calls self.draw() in other thread with render context (Qt)
            time.sleep(speed)  # parameter can be used to step slower.
            steps += 1

        # Without resetting velocity the reward of the next state is influenced by the current state.
        # With the resetting, it's a Markovian decisionproces.
        self.robot_sim.reset_velocity()
        # print("Steps " + str(steps) + " in " + str(round(end-start, 4)) + " Sekunden")

        reward = (self.robot_sim.get_body_pos()[0] - robot_start_pos[0])# - (self.get_steps_done() * 0.21)
        # self.print_vel()

        # Daten zur Auswertung sammeln

        # self.robot_sim.collect_angles()
        # self.robot_sim.collect_states()

        # if isinstance(self.robot, RobotDiscrete):
        #     if abs(reward) < self.reward_cap:  # cap reward to not learn from noise
        #         reward = 0

        # logging.debug("State: {}".format(self.robot.state))
        # logging.debug("Reward: {}".format(reward))

        return reward

    def get_steps_done(self):
        return self.robot_sim.get_steps_done()

    def draw_steps(self):
        self.robot_sim.draw_steps()

    def draw_angles(self):
        self.robot_sim.draw_angles()

    def draw_states(self):
        self.robot_sim.draw_states()

    def reset_robot_sim(self):
        self.robot_sim.reset()

    def get_mean_velocity(self):
        return sum(self.velocities) / len(self.velocities)

    def reset(self):
        """
        Resets the world.
        :return: state as one-hot encoding
        """
        pass
