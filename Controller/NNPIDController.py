from Controller.AbstractController import AbstractController
import jax.numpy as jnp
import numpy as np

class NNPIDController(AbstractController):

    def __init__(self, hidden_layers, min_init, max_init, input_d = 3, output_d = 1, dt=1):

        self.min_init = min_init
        self.max_init = max_init
        self.dt = dt
        
        layer_params = []
        activation_funcs = []
        if len(hidden_layers) == 0:
            layer_params.append(self.get_layer(input_d, output_d))
        else:
            for i in range(len(hidden_layers)):
                if i == 0: layer_params.append(self.get_layer(input_d, hidden_layers[i][0]))
                else: layer_params.append(self.get_layer(hidden_layers[i-1][0], hidden_layers[i][0]))
                activation_funcs.append(NNPIDController.get_activation_function(hidden_layers[i][1]))
            layer_params.append(self.get_layer(hidden_layers[-1][0], output_d))

        self.model_params = layer_params
        self.activation_funcs = activation_funcs
        

    def step(self, model_params, err_hist):
        E = err_hist[-1]
        dEdt = (E - err_hist[-2]) / self.dt if len(err_hist) > 1 else 0
        sum_E = sum(err_hist) * self.dt
        X = jnp.array([[E, sum_E, dEdt]])

        for i in range(len(model_params)-1):
            # print('SHAPES:', X.shape, model_params[i][0].shape)
            X = X @ model_params[i][0] + model_params[i][1]
            X = self.activation_funcs[i](X)

        # print('SHAPES:', X.shape, model_params[-1][0].shape)
        y = X @ model_params[-1][0] + model_params[-1][1]

        return jnp.squeeze(y)
    
    def get_layer(self, fan_in, fan_out):
        A = np.random.uniform(self.min_init, self.max_init, (fan_in, fan_out))
        b = np.random.uniform(self.max_init, self.max_init, (fan_out))
        return (A, b)


    @staticmethod
    def get_activation_function(name):
        match name.upper():
            case 'RELU': return NNPIDController.ReLU
            case 'SIGMOID': return NNPIDController.sigmoid
            case 'TANH': return NNPIDController.tanh
            case 'LINEAR': return NNPIDController.linear
            case _: return NNPIDController.linear
    
    @staticmethod
    def ReLU(x):
        return jnp.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1+jnp.exp(-x))
        
    @staticmethod
    def tanh(x):
        return (jnp.exp(2*x) - 1) / (jnp.exp(2*x) + 1)
    
    @staticmethod
    def linear(x):
        return x