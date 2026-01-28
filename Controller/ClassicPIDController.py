from Controller.AbstractController import AbstractController
import jax.numpy as jnp

class ClassicPIDController(AbstractController):

    def __init__(self):
        self.model_params = jnp.array([0.0, 0.0, 0.0])

    def step(self, model_params, E, IE, dE):
        E_vec = jnp.array([E, IE, dE])
        U = model_params @ E_vec
        return U
        
    