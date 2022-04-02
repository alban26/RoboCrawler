import string
import sys
import re

from PyQt5 import (QtGui, QtCore)
from PyQt5.QtGui import (QColor)
from PyQt5.QtCore import Qt

from Box2D import (b2AABB, b2CircleShape, b2Color, b2DistanceJoint,
                   b2EdgeShape, b2LoopShape, b2MouseJoint, b2Mul,
                   b2PolygonShape, b2PulleyJoint, b2Vec2)
from Box2D import (b2_pi, b2_staticBody, b2_kinematicBody, b2DrawExtended)


"""
Code is adapted from the official pybox2d example:
https://github.com/pybox2d/pybox2d/blob/master/examples/backends/pyqt4_framework.py
"""


class Pyqt5Draw(b2DrawExtended):
    """
    This debug drawing class differs from the other frameworks.  It provides an
    example of how to iterate through all the objects in the world and
    associate (in PyQt5's case) QGraphicsItems with them.

    While DrawPolygon and DrawSolidPolygon are not used for the core shapes in
    the world (DrawPolygonShape is), they are left in for compatibility with
    other frameworks and the tests.

    world_coordinate parameters are also left in for compatibility.  Screen
    coordinates cannot be used, as PyQt5 does the scaling and rotating for us.

    If you utilize this framework and need to add more items to the
    QGraphicsScene for a single step, be sure to add them to the temp_items
    array to be deleted on the next draw.
    """
    MAX_TIMES = 20
    axisScale = 0.4

    def __init__(self, framework, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.test = framework
        self.window = self.test.window
        self.scene = self.window.scene
        self.view = self.window.graphicsView
        self.item_cache = {}
        self.temp_items = []
        self.status_font = QtGui.QFont("Times", 10, QtGui.QFont.Bold)
        self.font_spacing = QtGui.QFontMetrics(self.status_font).lineSpacing()
        self.draw_idx = 0

    def StartDraw(self):
        for item in self.temp_items:
            self.scene.removeItem(item)
        self.temp_items = []

    def EndDraw(self):
        pass

    def SetFlags(self, **kwargs):
        """
        For compatibility with other debug drawing classes.
        """
        pass

    def DrawStringAt(self, x, y, str, color=None):
        item = QtGui.QGraphicsSimpleTextItem(str)
        if color is None:
            color = (255, 255, 255, 255)

        brush = QtGui.QBrush(QColor(255, 255, 255, 255))
        item.setFont(self.status_font)
        item.setBrush(brush)
        item.setPos(self.view.mapToScene(x, y))
        item.scale(1. / self.test.viewZoom, -1. / self.test.viewZoom)
        self.temp_items.append(item)

        self.scene.addItem(item)

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.test.viewZoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        line1 = self.scene.addLine(aabb.lowerBound.x, aabb.lowerBound.y,
                                   aabb.upperBound.x, aabb.lowerBound.y,
                                   pen=QtGui.QPen(QColor(*color.bytes)))
        line2 = self.scene.addLine(aabb.upperBound.x, aabb.upperBound.y,
                                   aabb.lowerBound.x, aabb.upperBound.y,
                                   pen=QtGui.QPen(QColor(*color.bytes)))
        self.temp_items.append(line1)
        self.temp_items.append(line2)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        line = self.scene.addLine(p1[0], p1[1], p2[0], p2[1],
                                  pen=QtGui.QPen(QColor(*color.bytes)))
        self.temp_items.append(line)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        p1 = xf.position
        p2 = p1 + self.axisScale * xf.R.x_axis
        p3 = p1 + self.axisScale * xf.R.y_axis

        line1 = self.scene.addLine(p1[0], p1[1], p2[0], p2[1],
                                   pen=QtGui.QPen(QColor(255, 0, 0)))
        line2 = self.scene.addLine(p1[0], p1[1], p3[0], p3[1],
                                   pen=QtGui.QPen(QColor(0, 255, 0)))
        self.temp_items.append(line1)
        self.temp_items.append(line2)

    def DrawCircle(self, center, radius, color, drawwidth=1, shape=None):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        border_color = [c * 255 for c in color] + [255]
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        pen.setWidth(0)
        ellipse = self.scene.addEllipse(center[0] - radius, center[1] - radius,
                                        radius * 2, radius * 2, pen=pen)
        self.temp_items.append(ellipse)

    def DrawSolidCircle(self, center, radius, axis, color, shape=None):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        ellipse = self.scene.addEllipse(center[0] - radius, center[1] - radius,
                                        radius * 2, radius * 2, brush=brush,
                                        pen=pen)
        line = self.scene.addLine(center[0], center[1],
                                  (center[0] - radius * axis[0]),
                                  (center[1] - radius * axis[1]),
                                  pen=QtGui.QPen(QColor(255, 0, 0)))

        self.temp_items.append(ellipse)
        self.temp_items.append(line)

    def DrawPolygon(self, vertices, color, shape=None):
        """
        Draw a wireframe polygon given the world vertices vertices (tuples)
        with the specified color.
        """
        poly = QtGui.QPolygonF()
        pen = QtGui.QPen(QtGui.QColor(*color.bytes))
        pen.setWidth(0)

        for v in vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, pen=pen)
        self.temp_items.append(item)

    def DrawSolidPolygon(self, vertices, color, shape=None):
        """
        Draw a filled polygon given the world vertices vertices (tuples) with
        the specified color.
        """
        poly = QtGui.QPolygonF()
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        pen.setWidth(0)

        for v in vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, brush=brush, pen=pen)
        self.temp_items.append(item)

    def DrawCircleShape(self, shape, transform, color, temporary=False):
        center = b2Mul(transform, shape.pos)
        radius = shape.radius
        axis = transform.R.x_axis

        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        pen.setWidth(0)
        ellipse = self.scene.addEllipse(-radius, -radius,
                                        radius * 2, radius * 2, brush=brush,
                                        pen=pen)
        pen1 = QtGui.QPen(QColor(255, 0, 0))
        pen1.setWidth(0)
        line = self.scene.addLine(center[0], center[1],
                                  (center[0] - radius * axis[0]),
                                  (center[1] - radius * axis[1]),
                                  pen=pen1)
        ellipse.setPos(*center)
        ellipse.radius = radius

        if temporary:
            self.temp_items.append(ellipse)
            self.temp_items.append(line)
        else:
            self.item_cache[hash(shape)] = [ellipse, line]

    def DrawPolygonShape(self, shape, transform, color, temporary=False):
        poly = QtGui.QPolygonF()
        border_color = color.bytes + [255]
        inside_color = (color / 2).bytes + [127]
        brush = QtGui.QBrush(QtGui.QColor(*inside_color))
        pen = QtGui.QPen(QtGui.QColor(*border_color))
        pen.setWidth(0)

        for v in shape.vertices:
            poly += QtCore.QPointF(*v)

        item = self.scene.addPolygon(poly, brush=brush, pen=pen)
        item.setRotation(transform.angle * 180.0 / b2_pi)
        item.setPos(*transform.position)

        if temporary:
            self.temp_items.append(item)
        else:
            self.item_cache[hash(shape)] = [item]

    def _clear_cache(self):
        for i in range(len(self.item_cache)):
            for item in self.item_cache.popitem()[1]:
                self.scene.removeItem(item)

    def _remove_from_cache(self, shape):
        items = self.item_cache[hash(shape)]

        del self.item_cache[hash(shape)]
        for item in items:
            self.scene.removeItem(item)

    def DrawShape(self, shape, transform, color, selected=False):
        """
        Draw any type of shape
        """
        cache_hit = False
        if hash(shape) in self.item_cache:
            cache_hit = True
            items = self.item_cache[hash(shape)]
            items[0].setRotation(transform.angle * 180.0 / b2_pi)
            if isinstance(shape, b2CircleShape):
                radius = shape.radius
                if items[0].radius == radius:
                    center = b2Mul(transform, shape.pos)
                    items[0].setPos(*center)
                    line = items[1]
                    axis = transform.R.x_axis
                    line.setLine(center[0], center[1],
                                 (center[0] - radius * axis[0]),
                                 (center[1] - radius * axis[1]))
                else:
                    self._remove_from_cache(shape)
                    cache_hit = False
            else:
                items[0].setPos(*transform.position)

            if not selected or cache_hit:
                return

        if selected:
            color = b2Color(1, 1, 1)
            temporary = True
        else:
            temporary = False

        if isinstance(shape, b2PolygonShape):
            self.DrawPolygonShape(shape, transform, color, temporary)
        elif isinstance(shape, b2EdgeShape):
            v1 = b2Mul(transform, shape.vertex1)
            v2 = b2Mul(transform, shape.vertex2)
            self.DrawSegment(v1, v2, color)
        elif isinstance(shape, b2CircleShape):
            self.DrawCircleShape(shape, transform, color, temporary)
        elif isinstance(shape, b2LoopShape):
            vertices = shape.vertices
            v1 = b2Mul(transform, vertices[-1])
            for v2 in vertices:
                v2 = b2Mul(transform, v2)
                self.DrawSegment(v1, v2, color)
                v1 = v2

    def DrawJoint(self, joint):
        """
        Draw any type of joint
        """
        bodyA, bodyB = joint.bodyA, joint.bodyB
        xf1, xf2 = bodyA.transform, bodyB.transform
        x1, x2 = xf1.position, xf2.position
        p1, p2 = joint.anchorA, joint.anchorB
        color = b2Color(0.5, 0.8, 0.8)

        if isinstance(joint, b2DistanceJoint):
            self.DrawSegment(p1, p2, color)
        elif isinstance(joint, b2PulleyJoint):
            s1, s2 = joint.groundAnchorA, joint.groundAnchorB
            self.DrawSegment(s1, p1, color)
            self.DrawSegment(s2, p2, color)
            self.DrawSegment(s1, s2, color)

        elif isinstance(joint, b2MouseJoint):
            pass  # don't draw it here
        else:
            self.DrawSegment(x1, p1, color)
            self.DrawSegment(p1, p2, color)
            self.DrawSegment(x2, p2, color)

    def reset(self):
        self._clear_cache()

    def ManualDraw(self):
        """
        This implements code normally present in the C++ version, which calls
        the callbacks that you see in this class (DrawSegment, DrawSolidCircle,
        etc.).

        This is implemented in Python as an example of how to do it, and also a
        test.
        """
        colors = {
            'active': b2Color(0.5, 0.5, 0.3),
            'static': b2Color(0.5, 0.9, 0.5),
            'kinematic': b2Color(0.5, 0.5, 0.9),
            'asleep': b2Color(0.6, 0.6, 0.6),
            'default': b2Color(0.7, 0.7, 0.7),
        }

        settings = self.test.settings
        world = self.test.world.b2World

        if settings.drawShapes:
            for body in world.bodies:
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape

                    if not body.active:
                        color = colors['active']
                    elif body.type == b2_staticBody:
                        color = colors['static']
                    elif body.type == b2_kinematicBody:
                        color = colors['kinematic']
                    elif not body.awake:
                        color = colors['asleep']
                    else:
                        color = colors['default']

                    self.DrawShape(fixture.shape, transform,
                                   color, False)

        if settings.drawJoints:
            for joint in world.joints:
                self.DrawJoint(joint)

        # if settings.drawPairs
        #   pass

        if settings.drawAABBs:
            color = b2Color(0.9, 0.3, 0.9)
            # cm = world.contactManager
            for body in world.bodies:
                if not body.active:
                    continue
                transform = body.transform
                for fixture in body.fixtures:
                    shape = fixture.shape
                    for childIndex in range(shape.childCount):
                        self.DrawAABB(shape.getAABB(
                            transform, childIndex), color)

    def to_screen(self, point):
        """
        In here for compatibility with other frameworks.
        """
        return tuple(point)