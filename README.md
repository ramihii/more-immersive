# Immersive AR - project

A desktop software that tracks ARuco cubes and replaces them with 3D models. Supports multiple cubes, different models.

Cubes use ARuco markers, 6 per cube (in order) so if you generate markers use the 1-6 for first cube, 7-12 for second etc.

Cubes are replaced by 3D models that are provided as command-line parameters.

## Dependencies:
* Python 3: https://www.python.org/
* Open Asset Import Library (assimp): https://github.com/assimp/assimp
* OpenCV for python: https://pypi.org/project/opencv-python/
* OpenCV contrib for python: https://pypi.org/project/opencv-contrib-python/

## Install
### Linux instructions

1. Make sure Python 3 is installed.

2. Install Open Asset Import Library (short: assimp).
It might be available via the system package manager.

On Arch Linux:
```
# pacman -S assimp
```

3. Install required python packages with pip:
```
# pip install pygame opencv-python pyopengl pyassimp opencv-contrib-python
```

4. Run 3d_viewer.py in its directory with:
```
$ python 3d_viewer.py tree.obj
```

### Windows instructions
1. Install Python 3

2. Install dependencies
```
# pip install pygame pyopengl opencv-python opencv-contrib-python
```

3. Install pyassimp
For x64 Windows the prebuilt dll don't work so you can download it from:
Download: https://drive.google.com/open?id=19286UTVZo4BLxf6Shm-q64kni7nFnnVI
run python setup.py install
copy assimp dll from win32/{ARCH} to {Python_Install_dir}/site-packages/pyassimp
ARCH is your Python 3 architecture (defaults to x86 when you download Python)
run python in commandline to find out

4. Run it
```
cd 3d_viewer
python 3d_viewer.py bottle.obj
```

## Use
### Calibration

1. Follow steps 1-3 of your system's setup instructions.

2. Print the camera calibration checkerboard pattern onto an A4 paper.
Other paper sizes can be used but the square size must be adjusted in
the script `opencv_arucu_sample/calibrate.py`.
The pattern can be found as a PDF file in
`opencv_arucu_sample/camera-calibration-checker-board_9x7.pdf`.

3. Take at least 10 images with the camera to be calibrated and place
them into the `opencv_arucu_sample/calibration-img/` directory.
The script `opencv_arucu_sample/calibration-img/camCap.py` can be used
to take the images.

4. Generate the `calibration.npz` file using the script
`opencv_arucu_sample/calibrate.py`. The script looks for PNG images by
default. If your images are in another format, change line 26 accordingly
in the `calibrate.py` script.

5. Copy the generated `calibration.npz` file to `3d_viewer/calibration.npz`.

### Marker generation

1. Follow steps 1-3 of your system's setup instructions.

2. Run the script `opencv_arucu_sample/generateMarkers.py` to generate the
marker images into the directory `opencv_arucu_sample/aruco/`. The directory
is created automatically if it doesn't exist.

3. Change the line 25 in the script `opencv_arucu_sample/aruco.py` to match
the size (in meters) of the generated marker(s).

### Running
start the program:
```
# python 3d_viewer.py <model1.obj, ...>
```
for example
```
# python 3d_viewer.py bottle.obj cup.obj
```
Use AR cubes to move the models.
```
ESC - to exit the program
```
