from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class AbstractPlant(ABC):
    
    def __init__(self, target_Y, min_D, max_D, dt=1):
        self.target_Y = target_Y
        self.D_min = min_D
        self.D_range = max_D - min_D
        self.dt = dt

    @abstractmethod
    def step(self, Y, U, D) -> Any :
        pass

    def get_disturbances(self, L):
        return np.random.random(L) * (self.D_range) + self.D_min
        
    def get_target_Y(self):
        return self.target_Y
