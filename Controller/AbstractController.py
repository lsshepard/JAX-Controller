from abc import ABC, abstractmethod
import jax.numpy as jnp

class AbstractController(ABC):
    
    @abstractmethod
    def step(self, E, model_params, err_hist) -> float:
        pass

    def get_model_params(self) -> jnp.ndarray:
        return self.model_params.copy()

    def set_model_params(self, new_params: jnp.ndarray):
        self.model_params = new_params.copy()