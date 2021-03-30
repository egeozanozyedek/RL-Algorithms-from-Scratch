import numpy as np
from src.NeuralNetwork.Layers.nn_toolkit import error_map


class Network(object):

    def __init__(self, layers, input_size, error_function):

        self.layers = layers
        self.input_size = input_size
        self.error_function = error_map[error_function]
        self.__initialize()



    def __initialize(self):

        next_size = self.input_size

        for layer in self.layers:
            next_size = layer.initialize(next_size)



    def fit(self, X, Y, epoch, learning_rate, momentum_rate, batch_size=None):

        loss_list = []

        for ep in range(epoch):

            pred, loss = self.__fit_instance(X, Y, learning_rate, momentum_rate)
            loss_list.append(loss)

        return pred, loss_list




    def __fit_instance(self, X, Y, learning_rate, momentum_rate):

        prediction = self.__call_forward(X)
        loss, residual = self.__calculate_error(prediction, Y)
        self.__call_backward(residual)
        self.__call_update(learning_rate, momentum_rate)

        return prediction, loss




    def predict(self, X):
        next_X = X

        for layer in self.layers:
            next_X = layer.predict(next_X)

        return next_X




    def __call_forward(self, X):

        next_X = X

        for layer in self.layers:
            next_X = layer.forward_pass(next_X)
        return next_X



    def __call_backward(self, residual):

        next_residual = residual

        for layer in reversed(self.layers):
            next_residual = layer.backward_pass(next_residual)




    def __call_update(self, learning_rate, momentum_rate):

        for layer in self.layers:
            layer.update(learning_rate, momentum_rate)



    def __calculate_error(self, pred, actual):
        error, residual = self.error_function(pred, actual)

        return error, residual



