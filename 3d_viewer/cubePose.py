import numpy as np
import cv2


dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

m = 0.047       # marker size in meters
b = 0.01       # border size in meters
n = 0.
c = m + b + b  # cube size
0
o = c / 2
h = m / 2
v = h + b
# 0 point in a corner
'''
boardCorners = [np.array([[b, b, n], [v, b, n], [v, v, n], [b, v, n]], dtype=np.float32),
                np.array([[b, c, b], [v, c, b], [v, c, v], [b, c, v]], dtype=np.float32),
                np.array([[b, n, v], [v, n, v], [v, n, b], [b, n, b]], dtype=np.float32),
                np.array([[b, v, c], [v, v, c], [v, b, c], [b, b, c]], dtype=np.float32),
                np.array([[n, b, v], [n, b, b], [n, v, b], [n, v, v]], dtype=np.float32),
                np.array([[c, b, b], [c, b, v], [c, v, v], [c, v, b]], dtype=np.float32)]

# 0 point in the middle of the cube
boardCorners = [np.array([[-h,-h, o], [-h, h, o], [ h, h, o], [ h,-h, o]], dtype=np.float32),
                np.array([[ o,-h, h], [ o, h, h], [ o, h,-h], [ o,-h,-h]], dtype=np.float32),
                np.array([[-o,-h,-h], [-o, h,-h], [-o, h, h], [-o,-h, h]], dtype=np.float32),
                np.array([[ h,-h,-o], [ h, h,-o], [-h, h,-o], [-h,-h,-o]], dtype=np.float32),
                np.array([[-h, o, h], [-h, o,-h], [ h, o,-h], [ h, o, h]], dtype=np.float32),
                np.array([[-h,-o,-h], [-h,-o, h], [ h,-o, h], [ h,-o,-h]], dtype=np.float32)]

boardIds = np.array([[0],[1],[2],[3],[4],[5]], dtype=np.int32)
board = cv2.aruco.Board_create(boardCorners, dic, boardIds)
'''

class Cube():
    boardCorners = [np.array([[-h, h, o], [ h, h, o], [ h,-h, o], [-h,-h, o]], dtype=np.float32),
                    np.array([[-o, h,-h], [-o, h, h], [-o,-h, h], [-o,-h,-h]], dtype=np.float32),
                    np.array([[ o, h, h], [ o, h,-h], [ o,-h,-h], [ o,-h, h]], dtype=np.float32),
                    np.array([[ h, h,-o], [-h, h,-o], [-h,-h,-o], [ h,-h,-o]], dtype=np.float32),
                    np.array([[-h,-o, h], [ h,-o, h], [ h,-o,-h], [-h,-o,-h]], dtype=np.float32),
                    np.array([[-h, o,-h], [ h, o,-h], [ h, o, h], [-h, o, h]], dtype=np.float32)]

    boardIds = np.array([[0],[1],[2],[3],[4],[5]], dtype=np.int32)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    
    def __init__(self, cubeId):
        self.cubeId = cubeId
        self.markerIds = 6*cubeId + self.boardIds 
        self.board = cv2.aruco.Board_create(self.boardCorners, self.d, self.markerIds)
        self.rvec = []
        self.tvec = []


    def detectCube(self, frame, mtx, dist, corners, ids):
        if ids is None:
            return [],[]
        ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, self.board, mtx, dist)
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 255))
        if rvec is None or tvec is None:
            return [],[]

        self.rvec = np.array( [[rvec[0][0], rvec[1], rvec[2]]] )
        self.tvec = np.array( [[tvec[0][0], tvec[1], tvec[2]]] )  
        return self.rvec, self.tvec

    def getRvec(self):
        return self.rvec

    def getTvec(self):
        return self.tvec


# Detects pose of the single cube, 
# Markers needs to be in correct order and orientotation to work
# currently only works with the boardIds 
def detectCube(frame, mtx, dist):
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dic)
    if ids is None:
        return [],[], frame
    ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, mtx, dist)
    
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 255))
    if rvec is None or tvec is None:
        return [],[], frame

    rvecs = np.array([[ [rvec[0][0], rvec[1], rvec[2]] ]] )
    tvecs = np.array([[ [tvec[0][0], tvec[1], tvec[2]] ]] )
    return rvecs, tvecs, frame


def drawAxis(frame, mtx, dist, rvec, tvec, corners, ids):    
    frame = cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)
    return frame


if __name__ == "__main__":
    with np.load('calibration.npz') as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    # webcam
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        ret, rvec, tvec, corners, ids = detectCube(frame, mtx, dist)
        if ret:
            frame = drawAxis(frame, mtx, dist, rvec, tvec, corners, ids)

        cv2.imshow("test", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
