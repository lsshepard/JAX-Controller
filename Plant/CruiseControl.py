import jax.numpy as jnp
import numpy as np
from Plant.AbstractPlant import AbstractPlant

class CruiseControl(AbstractPlant):

    def __init__(self, target_Y, min_D, max_D, drag_k=0.1, min_a=-np.pi/3, max_a=np.pi/3, dt=1):
        self.drag_k = drag_k
        self.min_a = min_a
        self.max_a = max_a
        super().__init__(target_Y, min_D, max_D, dt)
    
    def get_init_state(self):
        return self.get_target_Y(), 0

    def step(self, state, U, D):
        Y, angle = state
        new_angle = jnp.clip(angle + D, self.min_a, self.max_a)
        F_hill = 9.81 * jnp.sin(new_angle)
        F_drag = self.drag_k * Y
        a = U - F_hill - F_drag
        newY = jnp.clip(Y + a * self.dt, 0)
        return (newY, (newY, new_angle))
