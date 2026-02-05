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
        

    # entrypoint to train the controller
    def fit(self):

        # track stats
        start_time = time.time()
        self.iteration_hist = []
        self.mse_hist = []

        # jax traced function
        epoch_grads = jax.value_and_grad(self.run_epoch_mse, argnums=0)

        # jax compiled train step
        @jax.jit
        def train_step(model_params, disturbances):
            mse, grads = epoch_grads(model_params, disturbances)
            new_model_params = jax.tree.map(
                lambda p, g: p - self.lr * g,
                model_params,
                grads
            )
            return mse, new_model_params
        
        model_params = self.controller.get_model_params() # obtain original model parameters
        for i in range(self.num_epochs):
            if self.param_hist is not None: self.param_hist.append(model_params)    # track parameters
            disturbances = self.plant.get_disturbances(self.num_timesteps)          # sample disturbances before jax trace
            
            mse, model_params = train_step(model_params, disturbances)              # compute one epoch

            self.iteration_hist.append(i)                                           # track stats
            self.mse_hist.append(mse)

            if i+1 == self.num_epochs: print("Epoch", i+1, 'mse:', mse)             # log final mse

        self.controller.set_model_params(model_params)                              # save params when done training
        
        
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print('Training time:', elapsed_time)

    # wrapper around run_epoch to compute mse for jax
    def run_epoch_mse(self, model_params, disturbances):
        E_hist, _, _ = self.run_epoch(model_params, disturbances)
        MSE = jnp.mean(jnp.square(jnp.array(E_hist)))
        return MSE
    
    def run_epoch(self, model_params, disturbances):

        target_Y = self.plant.get_target_Y()

        # timestep function for jax scan
        def scan_step(carry, D):
            plant_state, U, IE, prev_E = carry

            # update plant
            Y, next_plant_state = self.plant.step(plant_state, U, D)

            # update errors
            E = target_Y - Y
            next_IE = IE + E
            dE = E - prev_E

            # update controller
            next_U = self.controller.step(model_params, E, next_IE, dE)

            # next state and tacking
            next_carry = (next_plant_state, next_U, next_IE, E)
            history = (E, Y, next_U)
            return next_carry, history
        
        init_carry = (self.plant.get_init_state(), 0.0, 0.0, 0.0)

        _, (E_hist, Y_hist, U_hist) = jax.lax.scan(scan_step, init_carry, disturbances)

        return E_hist, Y_hist, U_hist
    
    def visualize_run(self):
        disturbances = self.plant.get_disturbances(self.num_timesteps)
        model_params = self.controller.get_model_params()

        _, Y_hist, U_hist = self.run_epoch(model_params, disturbances)
        
        plt.plot(Y_hist, label='Y')
        plt.plot(U_hist, label='Signal')
        plt.plot(disturbances, label='Disturbance')
        plt.legend()
        plt.title('Trained control system')
        plt.xlabel('Timesteps')
        plt.ylabel('System values')
        plt.show()

    
    def visualize_training(self, param_labels=['Kp', 'Ki', 'Kd']):
        plt.plot(self.iteration_hist, self.mse_hist, label='MSE') # type: ignore
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        if self.param_hist:
            lines = plt.plot(self.iteration_hist, self.param_hist) # type: ignore
            [line.set_label(l) for line, l in zip(lines, param_labels)]
            plt.title('Parameter evolution')
            plt.xlabel('Epochs')
            plt.ylabel('Parameter values')
            plt.legend()
            plt.show()