import math
import random
from AbstractPlant import AbstractPlant

class BathtubPlant(AbstractPlant):

    def __init__(self, A, C, H, minD, maxD, dt=1):
        self.A = A
        self.C = C
        self.H = H
        self.Dmin = minD
        self.Drange = maxD - minD
        self.disturbaces = []
        self.dt = dt

    def get_V(self):
        return math.sqrt(2*9.81*self.H)
    
    def get_disturbance(self):
        D = random.random() * (self.Drange) + self.Dmin
        self.disturbaces.append(D)

    def step(self, U):
        D = self.get_disturbance()
        V = self.get_V()
        Q = V*self.C
        dHdt = U + D - Q
        dH = dHdt * self.dt
        self.H += dH
        return self.H
