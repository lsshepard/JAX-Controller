from Controller.ClassicPIDController import ClassicPIDController
from Plant.BathtubPlant import BathtubPlant
from Plant.CournotCompetition import CournotCompetition
from ControlSystem import ControlSystem


controller = ClassicPIDController()
# plant = BathtubPlant(10, -0.1, 0.1, 1, 0.1)
plant = CournotCompetition(1, -0.1, 0.1, 10, 0.5)
ConSys = ControlSystem(plant, controller, 100, 20, 0.000001, track_params=True)

ConSys.fit()
ConSys.visualize_training()
ConSys.visualize_run()