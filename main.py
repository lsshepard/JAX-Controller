from Controller.ClassicPIDController import ClassicPIDController
from Controller.NNPIDController import NNPIDController
from Plant.Bathtub import Bathtub
from Plant.CournotCompetition import CournotCompetition
from ControlSystem import ControlSystem


# controller = ClassicPIDController()
# plant = Bathtub(10, -0.2, 0.2, 1, 0.1)
# ConSys = ControlSystem(plant, controller, 150, 10, 0.0001, track_params=False)

controller = NNPIDController([(5, 'RELU'), (3, 'RELU')], 0, 1)
plant = Bathtub(10, -0.3, 0.3, 1, 0.1)
ConSys = ControlSystem(plant, controller, 100, 20, 0.00005, track_params=False)

# plant = CournotCompetition(1, -0.1, 0.1, 10, 0.5)
# ConSys = ControlSystem(plant, controller, 50, 20, 0.0000001, track_params=True)

ConSys.fit()
ConSys.visualize_training()
ConSys.visualize_run()