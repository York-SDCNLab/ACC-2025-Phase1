import time
from queue import Queue
from typing import List, Dict, Any

import numpy as np

from core.roadmap import ACCRoadMap
from core.qcar import LidarSLAM
from core.control import WaypointProcessor
from .filter import QCarEKF

DT: float = 0.2

def array_to_list_in_obs(obs: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val.tolist()
    return obs


class PathTracker:
    def __init__(self) -> None:
        self.gps: LidarSLAM = LidarSLAM() # slam system
        self.processor: WaypointProcessor = WaypointProcessor(
            auto_stop=True
        )
        self.roadmap: ACCRoadMap = ACCRoadMap()
        self.last_state: np.ndarray = np.zeros(6)

    def reset(self, node_sequence_1: List[int], node_sequence_2: List[int] = None) -> Dict[str, Any]:
        # Generate the total trajectory
        if node_sequence_2 is None:
            waypoints: np.ndarray = self.roadmap.generate_path(node_sequence_1)
            self.processor.set_destination(waypoints[-1])
        else:
            waypoints_1: np.ndarray = self.roadmap.generate_path(node_sequence_1)
            waypoints_2: np.ndarray = self.roadmap.generate_path(node_sequence_2)
            waypoints: np.ndarray = np.concatenate((waypoints_1, waypoints_2[:100]), axis=0)
            self.processor.set_destination(waypoints_1[-1])

        # Get the initial state
        for _ in range(5):
            self.last_state = self.__get_current_state()
            self.last_state[3:] = 0 # v, w, a should be 0
            time.sleep(0.2)
        # print("Initial State:", self.last_state)
        # Initialize the processor
        obs: Dict[str, Any] = self.processor.reset(self.last_state, {}, waypoints)

        return obs

    def step(self) -> Dict[str, Any]:
        current_state: np.ndarray = self.__get_current_state()
        if not np.array_equal(current_state[:3], self.last_state[:3]):
            self.last_state = current_state
        obs: Dict[str, Any] = self.processor.step(self.last_state, {'stop': False})

        return obs

    def terminate(self) -> None:
        self.gps.terminate()

    def get_tracking(self) -> Dict[str, Any]:
        return self.processor.waypoints

    def __get_current_state(self) -> np.ndarray:
        # Read state info
        self.gps.readGPS()
        x: float = self.gps.position[0]
        y: float = self.gps.position[1]
        yaw: float =  self.gps.orientation[2]
        # Cal linear veloclity
        vx: float = (x - self.last_state[0]) / DT
        vy: float = (y - self.last_state[1]) / DT
        v: float = np.hypot(vx, vy)
        # Cal angular velocity
        w: float = (yaw - self.last_state[2]) / DT
        # Cal acceleration
        a: float = (v - self.last_state[3]) / DT

        return np.array([x, y, yaw, v, w, a])
    

class PathTrackWrapper:
    def __init__(self) -> None:
        self.path_tracker: PathTracker = PathTracker()

    def reset(self, node_sequence: List[int], obs_queue: Queue) -> None:
        for i in range(5):
            start: float = time.perf_counter()
            # Get the initial observation      
            obs: Dict[str, Any] = self.path_tracker.reset(node_sequence)
            obs_queue.put(obs) # Send to mpc
            # Make sure the dt is 0.2
            elapsed: float = time.perf_counter() - start
            time.sleep(max(0, DT - elapsed))

    def step(self, obs_queue: Queue) -> None:
        start: float = time.perf_counter()
        # Get the current observation
        obs: Dict[str, Any] = self.path_tracker.step()
        if obs_queue.full():
            obs_queue.get()
        obs_queue.put(obs) # Send to mpc
        # Make sure the dt is 0.2
        elapsed: float = time.perf_counter() - start
        time.sleep(max(0, DT - elapsed))
