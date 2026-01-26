import numpy as np
import jax.numpy as jnp
import jax
from Plant.AbstractPlant import AbstractPlant

class BathtubPlant(AbstractPlant):

    def __init__(self, target_Y, min_D, max_D, A, C, dt=1):
        self.A = A
        self.C = C
        super().__init__(target_Y, min_D, max_D, dt)

    def get_V(self, H):
        return jnp.sqrt(2*9.81*jnp.maximum(H, 0))

    def step(self, Y, U, D):
        V = self.get_V(Y)
        Q = V*self.C
        dBdt = U + D - Q
        dYdt = dBdt / self.A
        dY = dYdt * self.dt
        newY = Y + dY
        return newY
