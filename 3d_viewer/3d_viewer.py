#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# The rest of our code is in here that's why we are namespace polluting
from render_model import *

import sys
import transformations as tf

import cv2
import numpy as np

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

# marker dictionary
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


# Load previously saved camera calibration data
# TODO we need to check properly if there is a calibration file and inform the user
# TODO In ideal world we would just open a calibration program
# TODO running the program without ARuco (and calibration), useful for testing
#   just contain the ARuco things (calibration, globals etc.) behind a function call
with np.load('calibration.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


# Return the orientation and translation vectors from ARuco
# TODO marker -> transformation map
#   since now we just take the first marker meaning we can't use two trackers
#   and rotating the cube will cause the object to jump around
# for starters this could be solved by having a global for the marker id
#   take first marker id when the program is run and track only that marker
# later we need to map six markers into one model and so on
def readAndDrawMarkers(frame):
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, d)

    # No markers
    if ids is None:
        return [],[];

    # highlight detected markers and draw ids
    # This modifies the original image
    image = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 255) )

    # rotation and translation vectors
    # 0.07 is marker size in meters
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers( corners, 0.07, mtx, dist )

    # draw axis for the markers
    # TODO do we want?
    #for i in range( len(ids)):
    #    image = cv2.aruco.drawAxis(image, mtx, dist, rvecs[i], tvecs[i], 0.05)

    return rvecs, tvecs

def main(models, width, height):
    app = PyAssimp3DViewer(models, w=width, h=height)

    clock = pygame.time.Clock()

    # Transform the whole model (root object)
    # Example translation matrix
    t = tf.translation_matrix((0, 0, 0))
    # or t = tf.identity_matrix()

    # Example rotation matrix
    # take a look at transformations.py for more
    origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    #Rx = tf.rotation_matrix(0*math.pi/180, xaxis)
    #Ry = tf.rotation_matrix(45*math.pi/180, yaxis)
    #Rz = tf.rotation_matrix(30*math.pi/180, zaxis)
    #R = tf.concatenate_matrices(Rx, Ry, Rz)
    R = tf.identity_matrix()
    # scale matrix
    sm = tf.scale_matrix(10, origin)

    cam = cv2.VideoCapture(0)

    while app.loop():
        # Video camera frame
        ret,frame = cam.read()

        # change frame to RGB
        tvecs = []
        rvecs, tvecs = readAndDrawMarkers(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Translation
        # TODO scale is all messed up
        # need to test properly with light conditions where the marker doesn't get lost
        # figure out if scale is constant or depends on the camera distance or smth?
        scale = 100
        if len(tvecs) > 0:
            v = tvecs[0][0]
            # negatives because our tracking camera is in front of us (webcam)
            t = tf.translation_matrix((-scale*v[0], -scale*v[1], scale*v[2]))
        # Rotation
        # TODO something wrong with the rotation, investigate
        if len(rvecs) > 0:
            r = rvecs[0][0]
            r[2] = -r[2]
            # transform rotation vector (r) to 4x4 OpenGL matrix
            # Rotation matrix is correct but the axes and directions may be wrong
            m, _ = cv2.Rodrigues(r)
            for i in range(0, 3):
                for j in range(0, 3):
                    R[i][j] = m[i][j]
            #print("R={}".format(R))


        # Use np.dot to combine to transformation matrices
        # scene.rootnode is the whole model you just imported
        # translation, scale, rotation
        for scene in app.scenes:
            scene.rootnode.transformation = np.dot(np.dot(t, sm), R)

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

    _, *args = sys.argv
    main(models=args, width=1024, height=768)

