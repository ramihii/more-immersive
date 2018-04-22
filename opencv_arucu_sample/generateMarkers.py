import cv2
import numpy
import os

# marker dictionary
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

directory = './aruco/'

# how many markers you want to generate
n = 10

# generate aruco markers
# idNro: id of the marker
# size: size in pixels
def makeMarker(idNro, size):
    im = cv2.aruco.drawMarker(d, idNro, size)
    name = directory + 'id' + str(idNro) + '.png'
    cv2.imwrite(name, im)

if not os.path.exists(directory):
        os.mkdir(directory)

for i in range(n):
    makeMarker(i, 200)
