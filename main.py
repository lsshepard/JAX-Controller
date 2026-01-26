from Controller.ClassicPIDController import ClassicPIDController
from Plant.BathtubPlant import BathtubPlant
from ControlSystem import ControlSystem


controller = ClassicPIDController()
plant = BathtubPlant(10, -0.1, 0.1, 1, 0.1)
ConSys = ControlSystem(plant, controller, 30, 30, 0.001, track_params=True)

ConSys.fit()
ConSys.visualize_training()
ConSys.visualize_run()