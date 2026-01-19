from AbstractController import AbstractController

class ClassicPIDController(AbstractController):

    def __init__(self, dt=1):
        self.errHistory = []
        self.kp = 0
        self.ki = 0
        self.kd = 0
        self.dt = dt

    def step(self, E):
        dEdt = E / (self.errHistory[-1] * self.dt)
        self.errHistory.append(E)
        sumE = sum(self.errHistory)
        U = self.kp * E + self.ki * sumE + self.kd * dEdt
        return U  
        