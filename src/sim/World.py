import logging
import time
from Box2D import *

from robot.RobotDiscrete import RobotDiscrete


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
        self.renderer = renderer
        self.settings = settings
        self.reward_cap = 1  # cap reward under 1 to 0
        self.robot = robot
        self.robot_sim = robot_sim
        self.b2World = b2World
        self.ex = ex
        self.steps = 0
        self.total_distance_old = 0
        self.use_real_robot = False

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
        groundBodyDef.position = (0, -4)
        groundBody = self.b2World.CreateBody(groundBodyDef)
        groundBox = b2PolygonShape(box=(2000, 4))
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

    def step_reward(self):
        """
        Steps and renders until goal state is reached.
        :return: reward
        """
        draw_robot, speed = self.get_params()

        robot_start_pos = self.robot_sim.get_body_pos()
        time_steps = 0
        counter = 0
        while True:  # Simulate until next/goal state of robot is reached
            if isinstance(self.robot, RobotDiscrete):
                if self.robot_sim.update():
                    break

            if counter % 40000 == 0:
                self.steps += self.robot_sim.joint_vertices_y_axes()

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
            time_steps += 1

        # Without resetting velocity the reward of the next state is influenced by the current state.
        # With the resetting, it's a Markovian decisionproces.
        self.robot_sim.reset_velocity()


        # step_length_reward = 0
        # if self.robot_sim.robot_model.arms_num > 1:
        #     step_length_reward = self.robot_sim.get_joint_distance() * 0.05
        #     # 0.05
        #     if self.robot_sim.joints_switched():
        #         step_count_reward = 0.05
        print("Steps done: " + str(self.steps))
        reward = (self.robot_sim.get_body_pos()[0] - robot_start_pos[0])
        if isinstance(self.robot, RobotDiscrete):
            if abs(reward) < self.reward_cap:  # cap reward to not learn from noise
                reward = 0

        logging.debug("State: {}".format(self.robot.state))
        logging.debug("Reward: {}".format(reward))

        return reward

    def reset(self):
        """
        Resets the world.
        :return: state as one-hot encoding
        """
        pass
