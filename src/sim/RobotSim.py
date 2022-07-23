import math

import matplotlib
import numpy
import numpy as np
from Box2D import *
import logging
from collections import deque

from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 13})
from matplotlib.pyplot import axis

from robot.RobotDiscrete import RobotDiscrete


class Joint:
    """
    Joint of the robots arms.
    """

    def __init__(self, joint_body, joint_n_1_joint_n_joint):
        self.joint_body = joint_body
        self.joint_n_1_joint_n_joint = joint_n_1_joint_n_joint

    def get_joint_between_joints(self):
        """
        Returns the box2d joint between the body the two bodies of the joints
        :return:
        """
        return self.joint_n_1_joint_n_joint

    def get_joint_body(self):
        """
        Returns the joint body
        :return:
        """
        return self.joint_body

    def get_all(self):
        """
        Return all joint parameters
        :return: all joint parameters
        """
        return [self.joint_body, self.joint_n_1_joint_n_joint]


class RobotSim:
    """
    Model in Box2D physic library of the simulated robot within the graphical interface.
    """

    def __init__(self, robot_model, joints_mass, joints_size, joints_friction, b2World):
        self.b2World = b2World
        self.init(robot_model, joints_mass, joints_size, joints_friction)

    def init(self, robot_model, joints_mass, joints_size, joints_friction):
        self.robot_model = robot_model

        pos = (0, 0.8)

        body_mass = 5
        self.body_size = (2.2, 0.8)
        body_friction = 1.0

        self.down_up_1 = False
        self.down_up_2 = False
        self.steps_done = 0
        self.arm_1_step_x_data = []
        self.arm_2_step_x_data = []
        self.arm_1_step_y_data = []
        self.arm_2_step_y_data = []

        self.arm1_j1_angles = []
        self.arm1_j2_angles = []
        self.arm2_j1_angles = []
        self.arm2_j2_angles = []

        self.arm1_j1_states = []
        self.arm1_j2_states = []
        self.arm2_j1_states = []
        self.arm2_j2_states = []

        wheel_mass = 1
        self.wheel_radius = 0.25
        wheel_friction = 0.1
        body_wheel_joint_margin_body = (0.5, 0)

        self.joints_mass = joints_mass
        self.joints_size = joints_size
        self.joints_friction = joints_friction
        self.max_degree_error = 0.005  # in radians #0.05

        # Margin to the upper right corner of the body
        body_arm1_joint_margin_body = (0.05, 0.05)

        # Robot body
        body_body_def = b2BodyDef()
        body_body_def.position = pos
        body_body_def.type = b2_dynamicBody
        self.body_body = self.b2World.CreateBody(body_body_def)
        body_box = b2PolygonShape(box=self.body_size)
        # categoryBits and maskBits is for not collide with itself
        body_box_fixture = b2FixtureDef(shape=body_box, categoryBits=0x0002, maskBits=0x0004)
        body_box_fixture.density = body_mass
        body_box_fixture.friction = body_friction
        self.body_body.CreateFixture(body_box_fixture)

        # Robot wheel
        wheel_body_def = b2BodyDef()
        wheel_body_def.position = self.body_body.position + \
                                  (-self.body_size[0] / 1 + body_wheel_joint_margin_body[0],
                                   -self.body_size[1] / 1 + body_wheel_joint_margin_body[1])
        wheel_body_def.type = b2_dynamicBody
        self.wheel_body = self.b2World.CreateBody(wheel_body_def)
        wheel_circle = b2CircleShape(radius=self.wheel_radius)
        wheel_circle_fixture = b2FixtureDef(shape=wheel_circle, categoryBits=0x0002, maskBits=0x0004)
        wheel_circle_fixture.density = wheel_mass
        wheel_circle_fixture.friction = wheel_friction
        self.wheel_body.CreateFixture(wheel_circle_fixture)

        self.body_wheel_joint = self.b2World.CreateRevoluteJoint(bodyA=self.body_body, bodyB=self.wheel_body,
                                                                 anchor=(
                                                                     -self.body_size[0] / 1 +
                                                                     body_wheel_joint_margin_body[0],
                                                                     body_wheel_joint_margin_body[1]))

        # Robot arms with joints
        self.arms = []
        for arm in range(self.robot_model.arms_num):
            joints = []
            for joint in range(self.robot_model.joints_per_arm_num):
                joint_body_def = b2BodyDef()
                if joint == 0:  # Joint with body?
                    joint_body_def.position = self.body_body.position + \
                                              (self.body_size[0] / 1 - body_arm1_joint_margin_body[0] +
                                               joints_size[joint][0] / 1 - joints_size[joint][1] / 1,
                                               self.body_size[1] / 1 - body_arm1_joint_margin_body[1])
                else:
                    joint_body_def.position = joints[joint - 1].joint_body.position = joints[
                                                                                          joint - 1].joint_body.position + \
                                                                                      (joints_size[joint - 1][0] / 1 -
                                                                                       joints_size[joint - 1][1] / 1 +
                                                                                       joints_size[joint][0] / 1 -
                                                                                       joints_size[joint][1] / 1, 0)

                joint_body_def.type = b2_dynamicBody
                joint_body = self.b2World.CreateBody(joint_body_def)

                joint_box = b2PolygonShape(box=joints_size[joint])
                joint_box_fixture = b2FixtureDef(shape=joint_box, categoryBits=0x0002, maskBits=0x0004)
                joint_box_fixture.density = joints_mass[joint]
                joint_box_fixture.friction = joints_friction[joint]
                joint_body.CreateFixture(joint_box_fixture)

                if joint == 0:  # Joint with body?
                    body_joint1_joint = self.b2World.CreateRevoluteJoint(
                        bodyA=joint_body, bodyB=self.body_body,
                        localAnchorA=(-(joints_size[joint][0] / 1 - joints_size[joint][1] / 1), 0),
                        localAnchorB=(self.body_size[0] / 1 - body_arm1_joint_margin_body[0],
                                      self.body_size[1] / 1 - body_arm1_joint_margin_body[1]),
                        lowerAngle=self.robot_model.state_range_rad_joints[joint][0],
                        upperAngle=self.robot_model.state_range_rad_joints[joint][1],
                        enableLimit=True,
                        enableMotor=True,
                        maxMotorTorque=100000.0,
                        motorSpeed=0.0)
                    joints.append(Joint(joint_body, body_joint1_joint))
                else:
                    joint_n_1_joint_n_joint = self.b2World.CreateRevoluteJoint(
                        bodyA=joint_body, bodyB=joints[joint - 1].joint_body,
                        localAnchorA=(-(joints_size[joint][0] / 1 - joints_size[joint][1] / 1), 0),
                        localAnchorB=(joints_size[joint - 1][0] / 1 - joints_size[joint - 1][1] / 1, 0),
                        lowerAngle=self.robot_model.state_range_rad_joints[joint][0],
                        upperAngle=self.robot_model.state_range_rad_joints[joint][1],
                        enableLimit=True,
                        enableMotor=True,
                        maxMotorTorque=100000.0,
                        motorSpeed=0.0)
                    joints.append(Joint(joint_body, joint_n_1_joint_n_joint))
            self.arms.append(joints)

        # logging.debug("RobotSim init: "
        #               "body_mass={}, body_size={}, body_friction={}, "
        #               "joints_mass={}, joints_size={}, joints_friction={}, "
        #               "wheel_mass={}, wheel_radiue={}, wheel_friction={}, "
        #               .format(body_mass, self.body_size, body_friction,
        #                       self.joints_mass, self.joints_size, self.joints_friction,
        #                       wheel_mass, self.wheel_radius, wheel_friction))

    def reset(self):
        """
        Removes box2D objects from world and adds init all again.
        :return:
        """
        for joint in self.get_joints():
            self.b2World.DestroyJoint(joint)

        for body in self.get_bodies():
            self.b2World.DestroyBody(body)

        self.init(self.robot_model, self.joints_mass, self.joints_size, self.joints_friction)

    def get_joints(self):
        """
        Get all box2d joints.
        :return:
        """
        joints = [self.body_wheel_joint]
        for arm in self.arms:
            for joint in arm:
                joints += [joint.get_joint_between_joints()]

        return joints

    def get_bodies(self):
        """
        Get all box2d body objects.
        :return:
        """
        bodies = [self.body_body, self.wheel_body]
        for arm in self.arms:
            for joint in arm:
                bodies += [joint.get_joint_body()]

        return bodies

    def get_objects(self):
        """
        Get all box2d objects to add to your space.
        :return: all box2d objects
        """
        objects = [self.body_body, self.wheel_body, self.body_wheel_joint]
        for arm in self.arms:
            for joint in arm:
                objects += [joint.get_all()]

        return objects

    def get_body_pos(self):
        """
        Get the middle position of the robots body as (x, y).
        :return: robots body position (x, y)
        """
        return self.body_body.position.x, self.body_body.position.y

    def opposite_signs(self, x, y):
        """
        Checks if signs of given numbers are different
        :param x: old value
        :param y: new value
        :return: True if change of sign occurs
        """
        x = int(round(x, 0))
        y = int(round(y, 0))
        return (x ^ y) < 0

    def get_joint_distance(self):
        """
        Calculates the vector length between the
        2nd revolute joints of the arms
        :return: vector length between 2nd revolute
        joint of arm1 and 2nd revolute joint of arm2
        """
        joint_2 = []
        for joints in self.arms:
            joint_2.append((joints[1].get_joint_between_joints().anchorB[0],
                            joints[1].get_joint_between_joints().anchorB[1]))
        x, y = numpy.subtract(joint_2[1], joint_2[0])
        return math.hypot(x, y)

    def step_counter_by_ground_touch(self):
        """
        Registers the y-Axes of the
        joint body of the arms
        and check whether a step is done
        and counts up in each iteration
        :return: number of steps
        """
        steps_done = 0
        for idx, joints in enumerate(self.arms):
            # Punkte im lokalen KS
            point_1 = joints[1].joint_body.fixtures[0].shape.vertices[1]
            point_2 = joints[1].joint_body.fixtures[0].shape.vertices[2]

            # Punkte im globalen KS
            p1_p2 = (joints[1].joint_body.GetWorldPoint(point_1),
                     joints[1].joint_body.GetWorldPoint(point_2))
            if idx == 0:  # first leg
                vertices_pair = []
                for point in p1_p2:  # left point and right point
                    vertices_pair.append(point[1])
                if self.down_up_1:
                    if self.lifted_off(vertices_pair):
                        self.down_up_1 = False
                        # steps_done += 1
                else:
                    if self.touched_down(vertices_pair):
                        self.down_up_1 = True
                        steps_done += 1
            else:  # second leg
                vertices_pair = []
                for point in p1_p2:  # left point and right point
                    vertices_pair.append(point[1])
                if self.down_up_2:
                    if self.lifted_off(vertices_pair):
                        self.down_up_2 = False
                        # steps_done += 1
                else:
                    if self.touched_down(vertices_pair):
                        self.down_up_2 = True
                        steps_done += 1
        return steps_done

    def lifted_off(self, vertices):
        return vertices[0] > 0.023 and vertices[1] > 0.023

    def touched_down(self, vertices):
        return vertices[0] < 0.02 or vertices[1] < 0.02

    def step_counter(self):
        """
        counts up the steps
        :return:
        """
        self.steps_done += self.step_counter_by_ground_touch()

    def get_steps_done(self):
        """
        Return the steps done and
        sets the steps back to 0
        :return: steps done
        """
        steps = self.steps_done
        self.steps_done = 0
        return steps

    def collect_steps(self):

        for idx, joints in enumerate(self.arms):
            # im lokalen KS
            # point_1 = joints[1].joint_body.fixtures[0].shape.vertices[1]
            point_2 = joints[1].joint_body.fixtures[0].shape.vertices[2]

            # im globalen KS
            p2 = joints[1].joint_body.GetWorldPoint(point_2)

            if idx == 0:
                self.arm_1_step_x_data.append(p2[0])
                self.arm_1_step_y_data.append(p2[1])

            else:
                self.arm_2_step_x_data.append(p2[0])
                self.arm_2_step_y_data.append(p2[1])

    def collect_angles(self):
        angles = self.robot_model.get_arm_states_degree()
        self.arm1_j1_angles.append(angles[0])
        self.arm1_j2_angles.append(angles[1])
        self.arm2_j1_angles.append(angles[2])
        self.arm2_j2_angles.append(angles[3])

    def collect_states(self):
        states = self.robot_model.get_state()
        self.arm1_j1_states.append(states[0][0])
        self.arm1_j2_states.append(states[0][1])
        self.arm2_j1_states.append(states[1][0])
        self.arm2_j2_states.append(states[1][1])

    def draw_steps(self):
        plt.xlabel("Zeitschritt")
        plt.ylabel("Schritthöhe in Meter")
        plt.plot(self.arm_1_step_y_data[35:], label='Bein 1')
        plt.plot(self.arm_2_step_y_data[35:], label='Bein 2')
        plt.legend()
        plt.savefig("steps.png")
        # plt.axis([0, 160, 0, 0.04])
        plt.show()
        # self.arm_1_step_x_data = []
        # self.arm_1_step_y_data = []
        # self.arm_2_step_x_data = []
        # self.arm_2_step_y_data = []

    def draw_angles(self):
        plt.xlabel("Zeitschritt")
        plt.ylabel("Winkel in Grad")
        plt.plot(self.arm1_j1_angles[20:], label='Bein 1 Gelenk 1')
        plt.plot(self.arm1_j2_angles[20:], label='Bein 1 Gelenk 2')
        plt.plot(self.arm2_j1_angles[20:], label='Bein 2 Gelenk 1')
        plt.plot(self.arm2_j2_angles[20:], label='Bein 2 Gelenk 2')
        plt.legend()
        plt.savefig("angles.png")
        plt.show()

    def draw_states(self):
        plt.xlabel("Zeitschritt")
        plt.ylabel("Zustand")
        plt.plot(self.arm1_j1_states[20:], label='Bein 1 Gelenk 1')
        plt.plot(self.arm1_j2_states[20:], label='Bein 1 Gelenk 2')
        plt.plot(self.arm2_j1_states[20:], label='Bein 2 Gelenk 1')
        plt.plot(self.arm2_j2_states[20:], label='Bein 2 Gelenk 2')
        plt.legend()
        plt.savefig("states.png")
        plt.show()

    def update(self):
        """
        Returns true if the robot reached the next state, otherwise updates the motor and returns false.
        :return: true if the robot reached the next state, false if the robot didn't reached the next state yet.
        """
        if isinstance(self.robot_model, RobotDiscrete):
            in_final_state = True

            for j, arm in enumerate(self.arms):
                for i, joint in enumerate(arm):
                    if i == 0:
                        joint_n_diff = self.robot_model.diff(
                            self.robot_model.diff(self.body_body.angle, joint.joint_body.angle),
                            self.robot_model.state_lookuptable_joints[i][self.robot_model.state[j, i]])
                    else:
                        joint_n_diff = self.robot_model.diff(
                            self.robot_model.diff(arm[i - 1].joint_body.angle, joint.joint_body.angle),
                            self.robot_model.state_lookuptable_joints[i][self.robot_model.state[j, i]])

                    # Update jointN's motor
                    if np.abs(joint_n_diff) > self.max_degree_error:
                        joint.joint_n_1_joint_n_joint.motorSpeed = - (np.sign(
                            joint_n_diff) * np.abs(joint_n_diff) * 0.3 + np.sign(joint_n_diff) * 0.1)
                    else:
                        joint.joint_n_1_joint_n_joint.motorSpeed = 0

                    if np.abs(joint_n_diff) > self.max_degree_error:
                        in_final_state = False

            return in_final_state

    def reset_velocity(self):
        """
        Sets the velocity and the angular_velocity of all objects to zero.
        :return:
        """
        self.body_body.linearVelocity = 0, 0
        self.body_body.angularVelocity = 0
        self.wheel_body.linearVelocity = 0, 0
        self.wheel_body.angularVelocity = 0
        for arm in self.arms:
            for joint in arm:
                joint.joint_body.linearVelocity = 0, 0
                joint.joint_n_1_joint_n_joint.linearVelocity = 0, 0
                joint.joint_body.angularVelocity = 0
                joint.joint_n_1_joint_n_joint.angularVelocity = 0

    def print_vel(self):
        print("body-linear", self.body_body.linearVelocity)
        print("body-angular", self.body_body.angularVelocity)
        print("wheel-linear", self.wheel_body.linearVelocity)
        print("wheel-angular", self.wheel_body.angularVelocity)

    def robot_body_size(self):
        """
        Returns body size as (width, height).
        :return: body size as (width, height)
        """
        return self.body_size
