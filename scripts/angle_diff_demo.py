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
from core.roadmap import CustomRoadMap
from control import PathTracker, Actuator, QCarEKF
from perception import WorldModel

def plan(taxi_regions: List[tuple], roadmap: CustomRoadMap) -> tuple:
    sequence = np.array([], dtype=np.int32)
    dropoff_indicator = np.array([], dtype=bool)
    
    updated_taxi_regions = []
    if len(taxi_regions) > 1:
        for i in range(len(taxi_regions) - 1):
            end = taxi_regions[i][1]
            next_start = taxi_regions[i+1][0]

            updated_taxi_regions.append(taxi_regions[i])
            updated_taxi_regions.append((end, next_start))

        taxi_regions = updated_taxi_regions
    
    taxi_regions.insert(0, (10, taxi_regions[0][0])) #add a taxi region from the hub to the first point
    taxi_regions.append((taxi_regions[-1][1], 10)) #add a taxi region back to the hub after the last dropoff

    for i, (pickup, dropoff) in enumerate(taxi_regions):
        _, short_path = roadmap.find_shortest_path(pickup, dropoff)
        short_path = short_path[::-1]

        if i != 0:
            dropoffs = np.zeros((short_path.shape[0] - 1), dtype=bool)
            dropoffs[0] = True
            dropoff_indicator = np.concatenate([dropoff_indicator, dropoffs])
        else:
            dropoffs = np.zeros((short_path.shape[0] - 1), dtype=bool)
            dropoff_indicator = np.concatenate([dropoff_indicator, dropoffs])

        if i != 0:
            sequence = np.concatenate([sequence, short_path[1:]])
        else:
            sequence = np.concatenate([sequence, short_path])

    return sequence, dropoff_indicator


class Agent:
    def __init__(self, roadmap: CustomRoadMap) -> None:
        self.camera: RGBDCamera = RGBDCamera()
        self.perception: WorldModel = WorldModel(device='cpu')
        checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
        self.perception.load_state_dict(checkpoint['model_state_dict'])
        self.perception.eval()        

        self.filter: QCarEKF = None
        self.tracker: PathTracker = PathTracker(roadmap)
        self.actuator: Actuator = Actuator()
        self.predicted_state: int = 0
        self.last_state: int = 0
        self.halt_signal: bool = False     
        self.halt_timer = None  

    def reset(self, node_sequence: List[int], dropoff_sequence: List[bool]) -> None:
        obs: Dict[str, Any] = self.tracker.reset(node_sequence, dropoff_sequence)
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
            self.halt_signal = self.predicted_state
            #print(f"Halt signal: {self.halt_signal}, predicted state: {self.predicted_state}")
            #print(f"last state: {self.last_state}, current state: {self.predicted_state}")

        # TODO: Temp way for the state machine
        # self.halt_signal = False
        if obs['stop']:
            self.actuator.set_led_strip_color((0, 0, 255))
            #self.actuator.halt_car(self.action[1], 3/10)
        elif self.halt_signal in [1, 2, 4]:
            if self.halt_signal == 1 and self.halt_timer is None:
                self.halt_timer = time.time()

            if self.halt_timer is not None and time.time() - self.halt_timer > 1.0:
                obs["stop"] = False
            else:
                obs["stop"] = True
            
            self.actuator.set_led_strip_color((255, 0, 0))
        else:
            self.actuator.set_led_strip_color((0, 255, 0))

        if self.halt_timer is not None and self.halt_signal in [0, 2, 3, 4]:
            self.halt_timer = None

        self.action = self.actuator.step(obs)            
        #    if self.predicted_state == 1 and self.last_state == 0: # Stop sign
        #        self.actuator.halt_car(self.action[1], 0)
        #    elif self.predicted_state == 2: # Traffic light
        #        self.actuator.halt_car(self.action[1], 0)
            #self.action = np.array([-10, 0], dtype=np.float32)
        
        self.last_state = self.predicted_state
        end: float = time.perf_counter() - start

        return obs['done']

    def terminate(self) -> None:
        self.actuator.halt_car()
        self.actuator.terminate()
        self.camera.terminate()
        self.tracker.terminate()

if __name__ == '__main__':
    roadmap: CustomRoadMap = CustomRoadMap()
    agent: Agent = Agent(roadmap)

    #define taxi pickup and dropoffs here
    pickup_dropoffs = [
        (20, 9),

        #add aditional pickup/dropoff points below
    ]

    node_sequence, dropoff_sequence = plan(pickup_dropoffs, roadmap)
    agent.reset(node_sequence, dropoff_sequence)
    done: bool = False

    try:
        while True:
            done: bool = agent.step()
            if done:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        agent.terminate()