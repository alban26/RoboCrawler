"""
Code is adapted from the official pybox2d example:
https://github.com/pybox2d/pybox2d/blob/master/examples/settings.py
"""


class Settings(object):
    """
    Default settings
    """

    # Physics options
    hz = 90.0
    velocityIterations = 8
    positionIterations = 3
    # Makes physics results more accurate (see Box2D wiki)
    enableWarmStarting = True
    enableContinuous = True     # Calculate time of impact
    enableSubStepping = False

    # Drawing
    drawStats = False
    drawShapes = True
    drawJoints = False
    drawCoreShapes = False
    drawAABBs = False
    drawOBBs = False
    drawPairs = False
    drawContactPoints = False
    maxContactPoints = 100
    drawContactNormals = False
    drawFPS = False
    drawMenu = False             # toggle by pressing F1
    drawCOMs = False            # Centers of mass
    pointSize = 2.5             # pixel radius for drawing points

    # Miscellaneous testbed options
    pause = False
    singleStep = False
    # run the test's initialization without graphics, and then quit (for
    # testing)
    onlyInit = False