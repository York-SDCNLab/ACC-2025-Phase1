import time 
import math

import numpy as np

class SteeringController:
    def __init__(self) -> None:
        self.max_steering: float = np.pi / 6
        self.k_p: float = 0
        self.k_i: float = 0
        self.k_d: float = 0

        self.error_integrate: float = 0
        self.error_prev: float = 0

    def reset(self, k_p: float = 0.1, k_i: float = 0.5, k_d: float = 0.0) -> None:
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.start: float = time.perf_counter()

    def step(self, psi: np.ndarray, psi_ref: np.ndarray, dt: float, obs: np.ndarray) -> float:
        angle_diff: float = (psi_ref - psi + np.pi) % (2 * np.pi) - np.pi
        error: float = angle_diff

        self.error_integrate = error * dt
        error_derivative: float = (error - self.error_prev) / dt
        obs['error'] = error        
        # print(f"psi: {psi}, psi_ref: {psi_ref}, error: {error}")
        pts = obs['next_waypoints'][13]
        if error == 0:
            if pts[1] > 0:
                steering: float = -np.pi/6
            elif pts[1] < 0:
                steering: float = np.pi/6
            else:
                steering: float = self.k_p * error + self.k_i * self.error_integrate + self.k_d * error_derivative
        else:        
            steering: float = self.k_p * error + self.k_i * self.error_integrate + self.k_d * error_derivative
        return np.clip(steering, -self.max_steering, self.max_steering)
    

class SpeedController:
    def __init__(self) -> None:
        self.max_pwm: float = 0.3
        
        self.k_p: float = 0
        self.k_i: float = 0
        self.k_d: float = 0

        self.error_integrate: float = 0
        self.error_prev: float = 0

    def reset(self, k_p: float = 0.1, k_i: float = 0.5, k_d: float = 0.0) -> None:
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

    def step(self, velocity: float, desired_velocity: float, dt: float) -> float:
        error: float = desired_velocity - velocity
        self.error_integrate += error * dt
        error_derivative: float = (error - self.error_prev) / dt
        pwm: float = self.k_p * error + self.k_i * self.error_integrate + self.k_d * error_derivative
        return np.clip(pwm, -self.max_pwm, self.max_pwm)
