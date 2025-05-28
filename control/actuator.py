import time
from queue import Queue
from typing import Dict, Any, List

import numpy as np

from core.qcar import PhysicalCar
from core.roadmap import get_waypoint_pose
from core.policies import PurePursuiteAdaptor
from .controllers import SteeringController2, SpeedController

def cal_lateral_error(obs: Dict[str, Any]) -> float:
    # pose: np.ndarray = [0, 0] # obs['state'][:2]
    # pt_1: np.ndarray = obs['waypoints'][0]
    # pt_2: np.ndarray = obs['waypoints'][1]
    pose: np.ndarray = obs['state'][:2]
    pt_1: np.ndarray = obs['global_waypoints'][0]
    pt_2: np.ndarray = obs['global_waypoints'][1]
    # print(pose, pt_1, pt_2)

    t: np.ndarray = pt_2 - pt_1 # tangent vector
    n: np.ndarray = np.array([-t[1], t[0]]) # normal vector
    n_hat: np.ndarray = n / np.linalg.norm(n) # unit normal vector

    v: np.ndarray = pose - pt_1 # vector from point 1 to car
    lateral_error: float = -np.dot(v, n_hat) # lateral distance from center line

    return lateral_error


class Actuator(PhysicalCar):
    def __init__(self) -> None:
        super().__init__()
        self.pid_steering: SteeringController2 = SteeringController2()
        self.pure_pursuit: PurePursuiteAdaptor = PurePursuiteAdaptor(1.0)
        self.pid_speed: SpeedController = SpeedController()
        self.init_flag: bool = True

        self.strip_color = (0, 255, 0)
        self.time_elapsed = 0.0

    def reset(self) -> np.ndarray:
        self.pid_steering.reset(k_p=0.5, k_i=35, k_d=0.425)
        self.pid_speed.reset(k_p=0.1, k_i=0.5, k_d=0.0)
        self.start: float = time.perf_counter()
        return np.zeros(2, dtype=np.float32)

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        dt: float = time.perf_counter() - self.start        
        if obs['done']:
            self.halt_car()
            obs['lateral_offset'] = cal_lateral_error(obs)
            self.start = time.perf_counter()
            return np.zeros(2, dtype=np.float32)

        lai_r1, lai_r2, lai_r3 = 35, 37, 3
        traj: np.ndarray = obs['next_waypoints']
        state: np.ndarray = np.asarray(obs['state'])
        
        psi = state[2]  
        traj_ref_1 = get_waypoint_pose(traj, lai_r1)[1][2]   
        traj_ref_2 = get_waypoint_pose(traj, lai_r2)[1][2]
        traj_ref_3 = get_waypoint_pose(traj, lai_r3)[1][2]
        if traj_ref_1 != traj_ref_2:
            lai, pwm,desired_vel = 52, 0.11, 1.1 # 0.09 # 0.08
        else:
            if abs(psi - traj_ref_3) <= 0.05:
                lai, pwm, desired_vel = 54, 0.15, 1.5 # 0.13
            else:
                lai, pwm, desired_vel = 52, 0.11, 1.1 # 0.09 # 0.08
        idx: int = lai if len(obs['next_waypoints']) > lai else len(obs['next_waypoints']) - 1
        psi_ref = get_waypoint_pose(traj, idx)[1][2]
        obs['psi'], obs['psi_ref'], obs['lateral_offset'] = psi, psi_ref, cal_lateral_error(obs)
        steering: float = self.pid_steering.step(obs, dt)

        # Actuate actions
        self.handle_leds(pwm, steering)        
        self.running_gear.read_write_std(pwm, steering, self.leds)
        self.time_elapsed += dt
        self.start = time.perf_counter()

        return np.array([pwm * 10, steering], dtype=np.float32)
    
    def set_led_strip_color(self, strip_color: tuple) -> None:
        self.strip_color = strip_color
    
    def get_hardware_metrics(self) -> float:
        self.running_gear.read()

        return np.concatenate([
            [self.running_gear.motorCurrent],
            [self.running_gear.batteryVoltage],
            self.running_gear.gyroscope,
            self.running_gear.accelerometer,
            self.running_gear.motorEncoder,
            [self.running_gear.motorTach]
        ])

class ActuatorWrapper:
    def __init__(self, throttle_coeff: float, steering_coeff: float) -> None:
        self.actuator: Actuator = Actuator(throttle_coeff, steering_coeff)

    def reset(self) -> None:
        self.actuator.reset()

    def step(self, action_queue: Queue) -> None:
        obs: Dict[str, Any] = action_queue.get()
        self.actuator.step(obs)

    def halt(self) -> None:
        self.actuator.halt_car()

    def terminate(self) -> None:
        self.actuator.terminate()