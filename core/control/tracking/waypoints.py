from typing import List, Tuple, Dict

import numpy as np

MAX_LOOKAHEAD_INDICES: int = 200

def is_in_area_aabb(state: np.ndarray, area_box: Dict[str, float]) -> bool:
    return area_box["min_x"] <= state[0] <= area_box["max_x"] and area_box["min_y"] <= state[1] <= area_box["max_y"]


class WaypointProcessor:
    """
    This class is used to process the waypoints and calculate the local waypoints for the agent.
    It also act as a pipeline to process the observation and add the waypoints info to it.

    Attributes:
    - task(List[int]): the task to be executed by the agent
    - waypoints(np.ndarray): the waypoints to be followed by the agent
    - ego_state(np.ndarray): the state of the ego car
    - current_waypoint_index(int): the index of the current waypoint
    - next_waypoints(np.ndarray): the next waypoints to be followed by the agent
    - local_waypoints(np.ndarray): the local waypoints to be followed by the agent
    - norm_dist(np.ndarray): the distance to the waypoints
    - dist_ix(int): the index of the closest waypoint
    - correction(float): the correction to be applied to the yaw angle
    - auto_stop(bool): a flag to determine if the agent should stop automatically when it reaches the last waypoint
    """

    def __init__(
        self, 
        max_lookahead_indices: int = MAX_LOOKAHEAD_INDICES, 
        use_optitrack: bool = False, 
        auto_stop: bool = False
    ) -> None:
        """
        The constructor for the WaypointProcessor class. It initializes some of the attributes.

        Parameters:
        - max_lookahead_indices(int): the maximum number of lookahead waypoints
        - use_optitrack(bool): a flag to determine if the agent should use the optitrack
        - auto_stop(bool): a flag to determine if the agent should stop automatically when it reaches the last waypoint
        """
        self.task: List[int] = None
        self.max_lookahead_indices: int = max_lookahead_indices
        self.local_waypoints: np.ndarray = np.zeros((self.max_lookahead_indices, 2))
        self.norm_dist: np.ndarray = None
        self.auto_stop: bool = auto_stop
        self.correction: float = np.pi if use_optitrack else 0 # we will not use lidar
        self.destination_area: Dict[str, float] = None

    def setup(
        self,
        ego_state: np.ndarray,
        observation: Dict[str, np.ndarray],
        waypoints: np.ndarray,
        init_waypoint_index: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        This method is used to setup the waypoint processor. It sets the ego state, waypoints, current waypoint index,
        next waypoints, local waypoints, distance to the waypoints, index of the closest waypoint, and correction to be
        applied to the yaw angle. It also adds the waypoints info to the observation.

        Parameters:
        - ego_state(np.ndarray): the state of the ego car
        - observation(Dict[str, np.ndarray]): the observation of the ego car
        - waypoints(np.ndarray): the waypoints to be followed by the agent
        - init_waypoint_index(int): the index of the initial waypoint

        Returns:
        - observation(Dict[str, np.ndarray]): the observation of the ego car with the waypoints info
        """
        self.waypoints = waypoints
        self.current_waypoint_index: int = init_waypoint_index
        self.next_waypoints: np.ndarray = self.waypoints[self.current_waypoint_index:]
        self.prev_waypoints: np.ndarray = self.waypoints[:self.current_waypoint_index]
        self.global_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:self.max_lookahead_indices]
        self.handle_observation(ego_state, observation)
        return observation

    def cal_vehicle_state(self, ego_state: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        This method is used to calculate the vehicle state. It calculates the origin, yaw, and rotation matrix of the
        vehicle.

        Parameters:
        - ego_state(np.ndarray): the state of the ego car

        Returns:
        - orig(np.ndarray): the origin of the vehicle
        - yaw(float): the yaw angle of the vehicle
        - rot(np.ndarray): the rotation matrix of the vehicle
        """
        orig: np.ndarray = ego_state[:2]
        yaw: float = -ego_state[2] + self.correction
        rot: np.ndarray = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig, yaw, rot

    def cal_local_waypoints(self, orig: np.ndarray, rot: np.ndarray) -> None:
        """
        This method is used to calculate the local waypoints. It calculates the local waypoints, distance to the
        waypoints, index of the closest waypoint, and updates the current waypoint index.

        Parameters:
        - orig(np.ndarray): the origin of the vehicle
        - rot(np.ndarray): the rotation matrix of the vehicle

        Returns:
        - None
        """
        # get the global waypoints
        self.global_waypoints: np.ndarray = np.roll(
            self.waypoints, -self.current_waypoint_index, axis=0
        )[:self.max_lookahead_indices]
        # get the distance to the waypoints
        self.norm_dist: np.ndarray = np.linalg.norm(self.global_waypoints - orig, axis=1)
        # get the index of the closest waypoint
        self.dist_ix: int = np.argmin(self.norm_dist)
        # update the current waypoint index
        self.current_waypoint_index = (self.current_waypoint_index + self.dist_ix) % self.waypoints.shape[0]
        # clear pasted waypoints
        self.next_waypoints = self.next_waypoints[self.dist_ix:]
        self.prev_waypoints = self.waypoints[self.current_waypoint_index - self.max_lookahead_indices:self.current_waypoint_index]
        # in case of less than 200 waypoints
        if self.next_waypoints.shape[0] < self.max_lookahead_indices:
            slop = self.max_lookahead_indices - self.next_waypoints.shape[0]
            self.next_waypoints = np.concatenate([self.next_waypoints, self.waypoints[:slop]])
        # convert the global frame to the local frame
        self.local_waypoints = np.matmul(self.next_waypoints[:self.max_lookahead_indices] - orig, rot)
        # self.local_waypoints = np.matmul(rot, (self.next_waypoints[:self.max_lookahead_indices] - orig).T)

    def handle_observation(self, ego_state: np.ndarray, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        This function is used to add the waypoints info and the ego state to the observation.

        Parameters:
        - observation(Dict[str, np.ndarray]): the observation of the ego car

        Returns:
        - observation(Dict[str, np.ndarray]): the observation of the ego car with the waypoints info (in local frame)
        """
        observation['next_waypoints'] = self.next_waypoints.copy()
        observation['prev_waypoints'] = self.prev_waypoints.copy()
        observation['global_waypoints'] = self.global_waypoints.copy()
        observation['waypoints'] = self.local_waypoints.copy()
        observation['state'] = ego_state.copy()
        observation['done'] = is_in_area_aabb(ego_state, self.destination_area) if self.current_waypoint_index > 100 and self.auto_stop else False
        return observation

    def execute(self, ego_state: np.ndarray, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        The main method of the WaypointProcessor class. It calculates the local waypoints, distance to the waypoints,
        index of the closest waypoint, and updates the current waypoint index. It also adds the waypoints info to the
        observation.

        Parameters:
        - ego_state(np.ndarray): the state of the ego car
        - observation(Dict[str, np.ndarray]): the observation of the ego car

        Returns:
        - observation(Dict[str, np.ndarray]): the observation of the ego car with the waypoints info
        """
        if ego_state is not None:
            orig, _, rot = self.cal_vehicle_state(ego_state)
            self.cal_local_waypoints(orig, rot)
            return self.handle_observation(ego_state, observation)
        
    def set_destination(self, destination: np.ndarray) -> None:
        self.destination_area = {
            'max_x': destination[0] + 0.2,
            'min_x': destination[0] - 0.2,
            'max_y': destination[1] + 0.2,
            'min_y': destination[1] - 0.2,
        }

    reset = setup
    step = execute