import sys
import time
import warnings
warnings.warn("This is just a prototype")
sys.path.insert(0, sys.path[0] + "/..")
from typing import List, Dict, Any

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.qcar import RGBDCamera
from control import PathTracker, Actuator, QCarEKF
from perception import WorldModel

def plot_trajectory(trajs: List[np.ndarray]) -> None:
    for i, traj in enumerate(trajs):
        x, y = traj.T
        plt.plot(x, y, label=f'Traj {i+1}')
    
    plt.title("Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectory.png")


class Agent:
    def __init__(self) -> None:
        self.camera: RGBDCamera = RGBDCamera()
        self.perception: WorldModel = WorldModel(device='cpu')
        checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
        self.perception.load_state_dict(checkpoint['model_state_dict'])
        self.perception.eval()        

        self.filter: QCarEKF = None
        self.tracker: PathTracker = PathTracker()
        self.actuator: Actuator = Actuator()
        self.predicted_state: int = 0
        self.halt_signal: bool = False       

    def reset(self, node_sequence: List[int]) -> None:
        obs: Dict[str, Any] = self.tracker.reset(node_sequence)
        self.model_state: tuple = self.perception.init_state()
        self.action: np.ndarray = self.actuator.reset()
        self.filter = QCarEKF(obs['state'][:3])
        self.last_filter_update: float = time.perf_counter()
        self.state = obs['state']

    def step(self) -> None:
        start: float = time.perf_counter()

        obs: Dict[str, Any] = self.tracker.step()
        hardware_metrics: np.ndarray = self.actuator.get_hardware_metrics()
        self.filter.update(
            u = self.action * np.array([1.1, 0.95]),
            dt = time.perf_counter() - self.last_filter_update,
            y_gps = obs['state'][:3],
            y_imu = hardware_metrics[0]
        )
        self.last_filter_update: float = time.perf_counter()
        self.state = np.concatenate([self.filter.x_hat[:, 0], hardware_metrics[[-1]], self.state[4:]])
        obs['state'] = self.state

        image: np.ndarray = self.camera.read_rgb_image()
        if image is not None:
            image = cv2.resize(image, (320, 240)).transpose(1, 0, 2)
            obs['image'], obs['action'] = image, self.action            
            pred, self.model_state, _ = self.perception.forward(obs.copy(), self.model_state)
            self.predicted_state: int = pred["fsm"].squeeze().cpu().numpy().argmax().item() 
            self.halt_signal = self.predicted_state in [1, 2, 4]
            print(f"Halt signal: {self.halt_signal}, predicted state: {self.predicted_state}")

        # TODO: Temp way for the state machine
        self.halt_signal = False
        if self.halt_signal:
            if self.predicted_state == 1: # Stop sign
                self.actuator.halt_car(self.action[1], 3)
            elif self.predicted_state == 2: # Traffic light
                self.actuator.halt_car(self.action[1])
            self.action = np.zeros(2, dtype=np.float32)
        else:
            self.action = self.actuator.step(obs)

        end: float = time.perf_counter() - start
        # time.sleep(max(0, 0.2 - end))

    def terminate(self) -> None:
        self.actuator.halt_car()
        self.actuator.terminate()
        self.camera.terminate()
        self.tracker.terminate()

if __name__ == '__main__':
    agent: Agent = Agent()
    # agent.reset([10, 1, 13, 17, 15, 6, 8] * 20) # Change the route here
    # agent.reset([10, 2, 4, 14, 22] * 20) # Change the route here
    agent.reset([10, 1, 17, 20, 22, 9, 7, 3, 1, 8] * 20) # Change the route here
    try:
        while True:
            agent.step()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        # following: np.ndarray = agent.tracker.get_tracking()
        # plot_trajectory([np.array(agent.local_history), following])
        agent.terminate()