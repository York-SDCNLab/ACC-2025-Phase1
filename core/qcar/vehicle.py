import time
from abc import abstractmethod
from typing import Union, Tuple

import numpy as np

from pal.products.qcar import QCar

# from core.sensor.sensor import VirtualCSICamera, VirtualRGBDCamera
from core.templates.base_policy import PolicyAdapter, BasePolicy
from .constants import QCAR_ACTOR_ID, ENCODER_COUNTS_PER_REV
from .constants import PIN_TO_SPUR_RATIO, WHEEL_RADIUS


class BaseCar:
    """
    The BaseCar class is an abstract class that defines the interface for both
    the physical and virtual cars
    """

    def __init__(self, throttle_coeff: float, steering_coeff: float) -> None:
        """
        Initializes the BaseCar object

        Parameters:
        - throttle_coeff: float: The throttle coefficient of the car
        - steering_coeff: float: The steering coefficient of the car

        Returns:
        - None
        """
        self.throttle_coeff: float = throttle_coeff
        self.steering_coeff: float = steering_coeff

    def setup(self, *args) -> None:
        """
        The setup function of the car, the subclasses will implement the function
        """
        ...

    def execute(self, *args) -> Tuple:
        """
        The execute method is an abstract method that executes the action of the car

        Parameters:
        - args: Any: The arguments of the car

        Returns:
        - Tuple: The output of the car
        """
        ...


class PhysicalCar(BaseCar):
    """
    The PhysicalCar class is a class that defines the interface for the physical car
    """

    def __init__(self, throttle_coeff: float = 0.3, steering_coeff: float = 0.5) -> None:
        """
        Initializes the PhysicalCar object

        Parameters:
        - throttle_coeff: float: The throttle coefficient of the car
        - steering_coeff: float: The steering coefficient of the car

        Returns:
        - None
        """
        super().__init__(throttle_coeff, steering_coeff)
        self.running_gear: QCar = QCar()
        self.leds = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def setup(self, *args) -> None:
        ...

    def terminate(self) -> None:
        self.running_gear.terminate()

    def handle_leds(self, throttle: float, steering: float) -> None:
        """
        The handle_leds method handles the LEDs of the car, when the
        car is turning or reversing, the corresponding LEDs will be turned on

        Parameters:
        - throttle: float: The throttle value of the car
        - steering: float: The steering value of the car

        Returns:
        - None
        """
        # steering indicator
        if steering > 0.3:
            self.leds[0] = 1
            self.leds[2] = 1
        elif steering < -0.3:
            self.leds[1] = 1
            self.leds[3] = 1
        else:
            self.leds = np.array([0, 0, 0, 0, 0, 0, self.leds[6], self.leds[7]])
        # reverse indicator
        if throttle < -0.02:
            self.leds[5] = 1
        elif -0.02 <= throttle < 0:
            self.leds[4] = 1

    def estimate_speed(self) -> float:
        """
        The estimate_speed method estimates the speed of the car based on the motor tach

        Returns:
        - float: The estimated speed of the car
        """
        return float(self.running_gear.motorTach)

    def halt_car(self, steering: float = 0.0, halt_time: float = 0.1) -> None:
        """
        Halts the QCar.

        Parameters:
        - steering (float): The steering value for the QCar.
        - halt_time (float): The halt time for the QCar.

        Returns:
        - None
        """
        # if halt_time >= 3:
        #     self.leds: np.ndarray = np.concatenate((self.leds[:6], [1, 1]))
        self.running_gear.read_write_std(throttle=-0.01, steering=steering, LEDs=self.leds)
        time.sleep(halt_time)
        # self.leds = np.concatenate((self.leds[:6], [0, 0]))
        # self.running_gear.read_write_std(throttle=0, steering=steering, LEDs=self.leds)

    reset = setup