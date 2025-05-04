import sys
import time
sys.path.insert(0, sys.path[0] + "/..")

from core.qcar import RGBDCamera

if __name__ == '__main__':
    camera = RGBDCamera()
    while True:
        rgb_image = camera.read_rgb_image()