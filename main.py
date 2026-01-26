from ClassicPIDController import ClassicPIDController
from BathtubPlant import BathtubPlant
from ControlSystem import ControlSystem


controller = ClassicPIDController()
plant = BathtubPlant(1, 0.1, 10, -0.1, 0.1)
ConSys = ControlSystem(plant, controller, 15, 50, 0.0001, track_params=True)

ConSys.fit()
ConSys.visualize_training()
ConSys.visualize_run()