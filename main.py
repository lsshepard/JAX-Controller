from Controller.ClassicPIDController import ClassicPIDController
from Controller.NNPIDController import NNPIDController
from Plant.Bathtub import Bathtub
from Plant.CournotCompetition import CournotCompetition
from Plant.CruiseControl import CruiseControl
from ControlSystem import ControlSystem


# controller = ClassicPIDController()
# plant = Bathtub(10, -0.2, 0.2, 1, 0.1)
# ConSys = ControlSystem(plant, controller, 150, 10, 0.0001, track_params=False)

# controller = NNPIDController([(5, 'RELU'), (3, 'RELU')], 0, 1)
# plant = Bathtub(10, -0.3, 0.3, 1, 0.1)
# ConSys = ControlSystem(plant, controller, 100, 20, 0.00005, track_params=False)

# controller = NNPIDController([(1, 'SIGMOID')], 0, 0)
# plant = CournotCompetition(1, -0.1, 0.1, 10, 0.5)
# ConSys = ControlSystem(plant, controller, 50, 30, 0.00001, track_params=False)

# plant = CournotCompetition(1, -0.1, 0.1, 10, 0.5)
# ConSys = ControlSystem(plant, controller, 50, 20, 0.0000001, track_params=True)

# controller = ClassicPIDController()
# controller = NNPIDController([(1, 'SIGMOID')], 0, 0)
# plant = CruiseControl(10, -0.2, 0.2, 0.1)
# ConSys = ControlSystem(plant, controller, 50, 100, 0.0000002, track_params=False)


# ConSys.fit()
# ConSys.visualize_training()
# ConSys.visualize_run()

def run(plant, controller, epochs, timesteps, lr, visualize_run=False):
    track_params = isinstance(controller, ClassicPIDController)
    ConSys = ControlSystem(plant, controller, epochs, timesteps, lr, track_params)
    ConSys.fit()
    ConSys.visualize_training()
    if visualize_run: ConSys.visualize_run()

# run(CruiseControl(10, -0.2, 0.2, 0.1), NNPIDController([(1, 'SIGMOID')], 0, 0), epochs=20, timesteps=100, lr=2e-6, visualize_run=True)
# run(CruiseControl(10, -0.01, 0.01, 0.1), ClassicPIDController(), epochs=20, timesteps=100, lr=1e-7, visualize_run=True)

run(Bathtub(10, -0.1, 0.1, 1, 0.1), ClassicPIDController(), epochs=50, timesteps=100, lr=1e-7, visualize_run=True)
