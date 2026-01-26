import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from AbstractPlant import AbstractPlant
from AbstractController import AbstractController


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

            new_model_params = model_params - self.lr * grads
            self.controller.set_model_params(new_model_params)

            self.iteration_hist.append(i)
            self.mse_hist.append(mse)
            if self.param_hist is not None: self.param_hist.append(new_model_params)
            print("Epoch", i, 'mse:', mse)

    def run_epoch_mse(self, model_params, disturbances):

        E_hist, Y_hist, U_hist = self.run_epoch(model_params, disturbances)
        
        MSE = jnp.square(jnp.array(E_hist)).mean()
        return MSE
    
    def run_epoch(self, model_params, disturbances):
        Y_hist = []
        U_hist = []
        E_hist = [0.0]
        target_Y = self.plant.get_target_Y()

        Y = target_Y
        U = 0
        for j in range(self.num_timesteps):
            Y = self.plant.step(Y, U, disturbances[j])
            E = target_Y - Y
            U = self.controller.step(E, model_params, E_hist)
            
            E_hist.append(E)
            Y_hist.append(Y)
            U_hist.append(U)
        
        return E_hist, Y_hist, U_hist
    
    def visualize_run(self):
        disturbances = self.plant.get_disturbances(self.num_timesteps)
        model_params = self.controller.get_model_params()

        _, Y_hist, U_hist = self.run_epoch(model_params, disturbances)
        
        plt.plot(Y_hist)
        plt.plot(U_hist)
        plt.plot(disturbances)
        plt.show()


    
    def visualize_training(self):
        plt.plot(self.iteration_hist, self.mse_hist) # type: ignore
        plt.show()
        if self.param_hist:
            plt.plot(self.iteration_hist, self.param_hist) # type: ignore
            plt.show()