import numpy as np
np.random.seed(42)

from Controller.ClassicPIDController import ClassicPIDController
from Controller.NNPIDController import NNPIDController
from Plant.Bathtub import Bathtub
from Plant.CournotCompetition import CournotCompetition
from Plant.CruiseControl import CruiseControl
from ControlSystem import ControlSystem


def run(plant, controller, epochs, timesteps, lr, visualize_run=False):
    track_params = isinstance(controller, ClassicPIDController)
    ConSys = ControlSystem(plant, controller, epochs, timesteps, lr, track_params)
    ConSys.fit()
    ConSys.visualize_training()
    if visualize_run: ConSys.visualize_run()

# run 1
# run(Bathtub(10, -0.1, 0.1, 1, 0.1), ClassicPIDController(), epochs=100, timesteps=120, lr=1e-8, visualize_run=True)

# run 2
# run(Bathtub(10, -0.1, 0.1, 1, 0.1),  NNPIDController([(5, 'RELU'), (2, 'RELU')], 0, 0), epochs=1000, timesteps=120, lr=1e-4, visualize_run=True)

# run 3
# run(CournotCompetition(6, -0.1, 0.1, 10, 0.5), ClassicPIDController(), epochs=500, timesteps=120, lr=1e-7, visualize_run=True)

# run 4
# run(CournotCompetition(6, -0.1, 0.1, 10, 0.5), NNPIDController([(5, 'TANH'), (2, 'SIGMOID')], 0, 0), epochs=500, timesteps=120, lr=3e-8, visualize_run=True)

# run 5
run(CruiseControl(5, -0.01, 0.01, drag_k=0.1, min_a=-np.pi/4, max_a=np.pi/4), ClassicPIDController(), epochs=1000, timesteps=1000, lr=5e-9, visualize_run=True)

# run 6
run(CruiseControl(5, -0.01, 0.01, drag_k=0.1, min_a=-np.pi/4, max_a=np.pi/4), NNPIDController([(3, 'RELU')], -0.1, 0.1), epochs=1000, timesteps=1000, lr=1e-6, visualize_run=True)
