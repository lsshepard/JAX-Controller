import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import time

from Plant.AbstractPlant import AbstractPlant
from Controller.AbstractController import AbstractController


class ControlSystem:
    
    def __init__(self, plant: AbstractPlant, controller: AbstractController, num_epochs, num_timesteps, lr, track_params=False):
        self.plant = plant
        self.controller = controller
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.lr = lr
        self.iteration_hist = None
        self.mse_hist = None
        self.param_hist =  [] if track_params else None
        

    def fit(self):
        start_time = time.time()

        self.iteration_hist = []
        self.mse_hist = []
        epoch_grads = jax.value_and_grad(self.run_epoch_mse, argnums=0)

        @jax.jit
        def train_step(model_params, disturbances):
            mse, grads = epoch_grads(model_params, disturbances)
            new_model_params = jax.tree.map(
                lambda p, g: p - self.lr * g,
                model_params,
                grads
            )
            return mse, new_model_params

        for i in range(self.num_epochs):
            model_params = self.controller.get_model_params()
            disturbances = self.plant.get_disturbances(self.num_timesteps)
            
            mse, new_model_params = train_step(model_params, disturbances)

            self.controller.set_model_params(new_model_params)

            self.iteration_hist.append(i)
            self.mse_hist.append(mse)
            if self.param_hist is not None: self.param_hist.append(new_model_params)
            # if (i+1) % 5 == 0: print("Epoch", i+1, 'mse:', mse)

        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print('Training time:', elapsed_time)

    def run_epoch_mse(self, model_params, disturbances):
        E_hist, _, _ = self.run_epoch(model_params, disturbances)
        MSE = jnp.mean(jnp.square(jnp.array(E_hist)))
        return MSE
    
    def run_epoch(self, model_params, disturbances):
        Y_hist, U_hist, E_hist = [], [], []
        target_Y = self.plant.get_target_Y()
        plant_state = self.plant.get_init_state()
        U = 0
        IE = 0

        for j in range(self.num_timesteps):
            Y, plant_state = self.plant.step(plant_state, U, disturbances[j])
            E = target_Y - Y
            IE += E
            dE = (E - E_hist[-1]) if E_hist else 0
            
            U = self.controller.step(model_params, E, IE, dE)
            
            E_hist.append(E)
            Y_hist.append(Y)
            U_hist.append(U)
        
        return E_hist, Y_hist, U_hist
    
    def visualize_run(self):
        disturbances = self.plant.get_disturbances(self.num_timesteps)
        model_params = self.controller.get_model_params()

        _, Y_hist, U_hist = self.run_epoch(model_params, disturbances)
        
        plt.plot(Y_hist, label='Y')
        plt.plot(U_hist, label='Signal')
        plt.plot(disturbances, label='Disturbance')
        plt.legend()
        plt.show()

    
    def visualize_training(self, param_labels=['Kp', 'Ki', 'Kd']):
        plt.plot(self.iteration_hist, self.mse_hist, label='MSE') # type: ignore
        plt.legend()
        plt.show()
        if self.param_hist:
            lines = plt.plot(self.iteration_hist, self.param_hist) # type: ignore
            [line.set_label(l) for line, l in zip(lines, param_labels)]
            plt.legend()
            plt.show()