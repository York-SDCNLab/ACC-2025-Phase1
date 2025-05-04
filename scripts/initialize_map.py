import sys
sys.path.insert(0, sys.path[0] + "/..")
raise Exception("This script is deprecated and will be removed in a future release. Please use the new scripts in the scripts folder.")

import numpy as np

from qvl.qlabs import QuanserInteractiveLabs
from core.roadmap import ACCRoadMap
from core.environment.builder import GeneralMapBuilder
from core.environment.director import GeneralDirector
from core.environment.constants import FULL_CONFIG as config
from core.environment.simulator import QLabSimulator
from core.qcar import LidarSLAM

def initialize_map() -> ACCRoadMap:
    qlabs: QuanserInteractiveLabs = QuanserInteractiveLabs()
    qlabs.open("localhost")

    builder: GeneralMapBuilder = GeneralMapBuilder(qlabs)
    director: GeneralDirector = GeneralDirector(builder, build_walls=True)
    sim: QLabSimulator = QLabSimulator([0, 0], False)
    sim.render_map(director, config)
    sim.reset_map()

    roadmap: ACCRoadMap = ACCRoadMap()
    pose: np.ndarray = roadmap.nodes[24].pose # Change the node index to the desired starting point
    sim.set_car_pos(0, location=[pose[0], pose[1], 0], orientation=[0, 0, pose[2]])

    return sim, pose


if __name__ == '__main__':
    sim, pose = initialize_map()