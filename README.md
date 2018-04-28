# Immersive AR - project

## Linux instructions

1. Make sure Python 3 is installed.

2. Install Open Asset Import Library (short: assimp).
It might be available via the system package manager.

On Arch Linux:
```Bash
# pacman -S assimp
```

3. Install required python packages with pip:
```Bash
# pip install pygame opencv-python pyopengl pyassimp opencv-contrib-python
```

4. Run 3d_viewer.py in its directory with:
```Bash
$ python 3d_viewer.py tree.obj
```

## Windows instructions
1. Install Python 3

2. Install dependencies
pip install pygame pyopengl opencv-python opencv-contrib-python

3. Install pyassimp
Download: https://drive.google.com/open?id=19286UTVZo4BLxf6Shm-q64kni7nFnnVI
run python setup.py install
copy assimp dll from win32/{ARCH} to {Python_Install_dir}/site-packages/pyassimp
ARCH is your Python 3 architecture (defaults to x86 when you download Python)
run python in commandline to find out

4. Test it
commandline:
cd 3d_viewer
python 3d_viewer.py tree.obj

