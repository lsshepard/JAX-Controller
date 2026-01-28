import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

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

        self.iteration_hist = []
        self.mse_hist = []
        epoch_grads = jax.value_and_grad(self.run_epoch_mse, argnums=0)

        for i in range(self.num_epochs):
            model_params = self.controller.get_model_params()
            disturbances = self.plant.get_disturbances(self.num_timesteps)
            mse, grads = epoch_grads(model_params, disturbances)

            # print(mse, grads)

            # new_model_params = model_params - self.lr * grads
            new_model_params = jax.tree.map(
                lambda p, g: p - self.lr * g,
                model_params,
                grads
            )
            self.controller.set_model_params(new_model_params)

            self.iteration_hist.append(i)
            self.mse_hist.append(mse)
            if self.param_hist is not None: self.param_hist.append(new_model_params)
            if (i+1) % 5 == 0: print("Epoch", i+1, 'mse:', mse)

    def run_epoch_mse(self, model_params, disturbances):

        E_hist, _, _ = self.run_epoch(model_params, disturbances)
        
        MSE = jnp.square(jnp.array(E_hist)).mean()
        return MSE
    
    def run_epoch(self, model_params, disturbances):
        Y_hist, U_hist, E_hist = [], [], []
        target_Y = self.plant.get_target_Y()
        plant_state = self.plant.get_init_state()
        U = 0

        for j in range(self.num_timesteps):
            Y, plant_state = self.plant.step(plant_state, U, disturbances[j])
            E = target_Y - Y
            E_hist.append(E)
            U = self.controller.step(model_params, E_hist)
            
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

    
    def visualize_training(self):
        plt.plot(self.iteration_hist, self.mse_hist, label='MSE') # type: ignore
        plt.legend()
        plt.show()
        if self.param_hist:
            plt.plot(self.iteration_hist, self.param_hist) # type: ignore
            plt.show()