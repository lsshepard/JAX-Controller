import numpy as np
import jax.numpy as jnp
import jax
from AbstractPlant import AbstractPlant

class BathtubPlant(AbstractPlant):

    def __init__(self, A, C, H, min_D, max_D, dt=1):
        self.A = A
        self.C = C
        self.target_H = H
        self.D_min = min_D
        self.D_range = max_D - min_D
        self.dt = dt

    def get_V(self, H):
        return jnp.sqrt(2*9.81*jnp.maximum(H, 0))
    
    def get_disturbances(self, L):
        return np.random.random(L) * (self.D_range) + self.D_min
        # return jax.random.uniform(key=random_key, shape=(L,), minval=self.D_min, maxval=self.D_max)

    def step(self, Y, U, D):
        V = self.get_V(Y)
        Q = V*self.C
        dBdt = U + D - Q
        dYdt = dBdt / self.A
        dY = dYdt * self.dt
        newY = Y + dY
        return newY
    
    def get_target_Y(self):
        return self.target_H
