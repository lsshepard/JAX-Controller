from AbstractController import AbstractController
import jax.numpy as jnp

class ClassicPIDController(AbstractController):

    def __init__(self, dt=1):
        self.model_params = jnp.array([1.0, 0.0, 0.0])
        self.dt = dt

    def step(self, E, model_params, err_hist):
        dEdt = (E - err_hist[-1]) / self.dt
        sum_E = sum(err_hist) * self.dt
        E_vec = jnp.array([E, sum_E, dEdt])
        U = model_params @ E_vec
        return U
        
    def get_model_params(self) -> jnp.ndarray:
        return self.model_params.copy()

    def set_model_params(self, new_params: jnp.ndarray):
        self.model_params = new_params.copy()