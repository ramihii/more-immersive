#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# The rest of our code is in here that's why we are namespace polluting
from render_model import *

import sys
import transformations as tf

import cv2

# Usage
# python 3d_viewer.py model_filename

# Editing
# The main function (run at the program start) is last in this file __main__
# it only calls the main function defined belower
# The OpenGL code, model loader etc. can be found in render_model
# For basic usage (moving the model around, swapping models etc.)
#   you should be fine by modifying main function with the examples

# TODO
# TODO need to figure out how to display video on the background
#   Example: https://gist.github.com/snim2/255151
# This is going to have problems with the rendering order etc.
# we need to render to the OpenGL canvas after clearing but before rendering the object model
# so our rendering looks like:
#       clear backbuffer
#       draw video
#       draw 3D
#       draw overlay (text/HUD)
#       swap buffers
# probably will require modification of the rendering code in render_model.py
#
# TODO keyboard events (and mouse for that matter) are in render_model.py
# move them here!

def main(model, width, height):
    app = PyAssimp3DViewer(model, w=width, h=height)

    clock = pygame.time.Clock()

    # Transform the whole model (root object)
    # Example translation matrix
    t = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    # Example rotation matrix
    # take a look at transformations.py for more
    origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    Rx = tf.rotation_matrix(0*math.pi/180, xaxis)
    Ry = tf.rotation_matrix(45*math.pi/180, yaxis)
    Rz = tf.rotation_matrix(30*math.pi/180, zaxis)
    R = tf.concatenate_matrices(Rx, Ry, Rz)

    # Use np.dot to combine to transformation matrices
    # scene.rootnode is the whole model you just imported
    app.scene.rootnode.transformation = np.dot(R,t)

    # this is true: scene == scene.rootnode.parent
    # would be way more logical if rootnode.parent == None
    logger.info('parent = {}'.format(app.scene == app.scene.rootnode.parent))

    cam = cv2.VideoCapture(0)

    while app.loop():
        # Video camera frame
        ret,frame = cam.read()

        # change frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #logger.info("frame shape = {}".format(frame.shape))
        # draw background
        app.draw_background(frame)

        # calculate camera matrix
        app.update_view_camera()

        #app.render_grid()

        ## Main rendering
        app.render()

        ## GUI text display
        """
        app.switch_to_overlay()
        app.showtext("Active camera: %s" % str(app.current_cam), 10, app.h - 30)
        if app.currently_selected:
            app.showtext("Selected node: %s" % app.currently_selected, 10, app.h - 50)
            pos = app.h - 70

            app.showtext("(%sm, %sm, %sm)" % (app.currently_selected.transformation[0, 3],
                                              app.currently_selected.transformation[1, 3],
                                              app.currently_selected.transformation[2, 3]), 30, pos)

        app.switch_from_overlay()
        """

        # Make sure we do not go over 30fps
        clock.tick(30)

    logger.info("Quitting! Bye bye!")


#########################################################################
#########################################################################

if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Usage: " + __file__ + " <model>")
        sys.exit(2)

    main(model=sys.argv[1], width=1024, height=768)

