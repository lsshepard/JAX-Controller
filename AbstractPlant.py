from abc import ABC, abstractmethod
from typing import Any
import jax.numpy as jnp

class AbstractPlant(ABC):
    
    @abstractmethod
    def step(self, Y, U, D) -> Any :
        pass

    def get_disturbances(self, L) -> jnp.ndarray:
        ...

    def get_target_Y(self) -> float:
        ...