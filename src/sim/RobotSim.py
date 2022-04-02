import numpy as np
from Box2D import *
import logging

from robot.RobotDiscrete import RobotDiscrete


class Joint():
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


class RobotSim():
    """
    Model in Box2D physic library of the simulated robot within the graphical interface.
    """

    def __init__(self, robot_model, joints_mass, joints_size, joints_friction, b2World):
        self.b2World = b2World
        self.init(robot_model, joints_mass, joints_size, joints_friction)

    def init(self, robot_model, joints_mass, joints_size, joints_friction):
        self.robot_model = robot_model

        pos = (0, 8)

        body_mass = 1.0
        self.body_size = (22.0, 8)
        body_friction = 1.0

        wheel_mass = 1.0
        self.wheel_radius = 2.5
        wheel_friction = 0.1
        body_wheel_joint_margin_body = (5, 0)

        self.joints_mass = joints_mass
        self.joints_size = joints_size
        self.joints_friction = joints_friction
        self.max_degree_error = 0.05  # in radians

        # Margin to the upper right corner of the body
        body_arm1_joint_margin_body = (0.5, 0.5)

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
                                                                 -self.body_size[0] / 1 + body_wheel_joint_margin_body[
                                                                     0],
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
                        maxMotorTorque=1000000.0,
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
                        maxMotorTorque=1000000.0,
                        motorSpeed=0.0)
                    joints.append(Joint(joint_body, joint_n_1_joint_n_joint))
            self.arms.append(joints)

        logging.debug("RobotSim init: "
                      "body_mass={}, body_size={}, body_friction={}, "
                      "joints_mass={}, joints_size={}, joints_friction={}, "
                      "wheel_mass={}, wheel_radiue={}, wheel_friction={}, "
                      .format(body_mass, self.body_size, body_friction,
                              self.joints_mass, self.joints_size, self.joints_friction,
                              wheel_mass, self.wheel_radius, wheel_friction))

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
                joint.joint_body.angularVelocity = 0

    def robot_body_size(self):
        """
        Returns body size as (width, height).
        :return: body size as (width, height)
        """
        return self.body_size
