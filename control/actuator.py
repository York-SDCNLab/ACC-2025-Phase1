import time
from queue import Queue
from typing import Dict, Any, List

import numpy as np

from core.qcar import PhysicalCar
from core.roadmap import get_waypoint_pose
from core.policies import PurePursuiteAdaptor
from .controllers import SteeringController, SpeedController
from quanser.devices import Aaaf5050McK12LED

def cal_lateral_error(obs: Dict[str, Any]) -> float:
    pose: np.ndarray = [0, 0] # obs['state'][:2]
    pt_1: np.ndarray = obs['waypoints'][0]
    pt_2: np.ndarray = obs['waypoints'][1]
    # print(pose, pt_1, pt_2)

    t: np.ndarray = pt_2 - pt_1 # tangent vector
    n: np.ndarray = np.array([-t[1], t[0]]) # normal vector
    n_hat: np.ndarray = n / np.linalg.norm(n) # unit normal vector

    v: np.ndarray = pose - pt_1 # vector from point 1 to car
    lateral_error: float = np.dot(v, n_hat) # lateral distance from center line

    return lateral_error


class Actuator(PhysicalCar):
    def __init__(self) -> None:
        super().__init__()
        self.pid_steering: SteeringController = SteeringController()
        self.pure_pursuit: PurePursuiteAdaptor = PurePursuiteAdaptor(1.0)
        self.pid_speed: SpeedController = SpeedController()
        self.init_flag: bool = True

        #try setting color to blue
        self.led_strip = Aaaf5050McK12LED()
        led_uri = "tcpip://localhost:18969" 
        self.led_strip.open(led_uri, 33)
        self.strip_color = (0, 255, 0)

        #for speed control
        self.desired_speed = 1.0
        self.prev_speed = 0.0

        self.time_elapsed = 0.0

    def reset(self) -> np.ndarray:
        self.pid_steering.reset(k_p=0.4, k_i=0.0, k_d=0.0)
        self.pid_speed.reset(k_p=0.1, k_i=0.5, k_d=0.0)
        # self.led_strip.writeColors(self.led_strip, 33)
        self.start: float = time.perf_counter()
        return np.zeros(2, dtype=np.float32)

    def step(self, obs: Dict[str, Any]) -> np.ndarray:
        dt: float = time.perf_counter() - self.start

        lai_r1, lai_r2, lai_r3 = 35, 37, 3
        # lai_r1, lai_r2 = 60, 65

        traj: np.ndarray = obs['next_waypoints']
        state: np.ndarray = np.asarray(obs['state'])
        lateral: float = cal_lateral_error(obs)
        
        psi = state[2]  
        traj_ref_1 = get_waypoint_pose(traj, lai_r1)[1][2]   
        traj_ref_2 = get_waypoint_pose(traj, lai_r2)[1][2]
        traj_ref_3 = get_waypoint_pose(traj, lai_r3)[1][2]
        if traj_ref_1 != traj_ref_2:
            lai, pwm, desired_vel = 52, 0.09, 1.0 # 0.08
            self.pid_steering.kp = 2.5
        else:
            if abs(psi - traj_ref_3) <= 0.05:
                lai, pwm, desired_vel = 54, 0.14, 1.5
                self.pid_steering.kp = 2.4
            else:
                lai, pwm, desired_vel = 52, 0.09, 1.0 # 0.08
                self.pid_steering.kp = 2.5
        # lai, pwm = 45, 0.15
        # lai, pwm = 50, 0.15
        # print(obs['state'])
        idx: int = lai if len(obs['next_waypoints']) > lai else len(obs['next_waypoints']) - 1
        psi_ref = get_waypoint_pose(traj, idx)[1][2]
        obs['psi'], obs['psi_ref'] = psi, psi_ref  
        # print(f"psi: {psi}, psi_ref: {psi_ref}")

        # print("Lateral error: ", lateral)
        # print(dt)

        if abs(lateral) >= 0.045: #and self.init_flag:
            # if self.init_flag:
            action, _ = self.pure_pursuit.execute(obs)
            steering: float = action[1] * 0.5
            # print("Change to pure pursuit")
        else:
            steering: float = self.pid_steering.step(psi, psi_ref, dt, obs)
            self.init_flag = False
            # print("Change to angle diff")
        # Actuate actions
        # pwm: float = 0.15

        #speed control
        #pwm 0.14 ~= 1.0m/s
        #pwm 0.09 ~= 0.6m/s
        observed_vel = state[3]
        if obs["stop"]:
            desired_vel = 0.0

        pwm = (desired_vel - observed_vel)*0.25

        #self.handle_leds(pwm, steering)   
        self.led_strip.writeColors(self.strip_color, 33)     
        self.running_gear.read_write_std(pwm, steering, self.leds)
        self.time_elapsed += dt
        self.start = time.perf_counter()

        return np.array([pwm, steering], dtype=np.float32)
    
    def set_led_strip_color(self, strip_color: tuple) -> None:
        self.strip_color = strip_color

    def halt_car(self, steering: float = 0.0, halt_time: float = 0.1) -> None:
        #self.handle_leds(0.0, steering)   
        self.led_strip.writeColors(self.strip_color, 33)
        self.running_gear.read_write_std(throttle=-0.0, steering=steering, LEDs=self.leds)

        time.sleep(halt_time)
    
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