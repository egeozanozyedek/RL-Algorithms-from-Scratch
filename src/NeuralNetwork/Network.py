import numpy as np
from src.NeuralNetwork.Layers.nn_toolkit import error_map


class Network(object):

    def __init__(self, layers, input_size, error_function, optimizer="sgd"):
        """
        Initializes important network parameters
        :param layers: Network layers
        :param input_size: input size of first layer and also the network
        :param error_function: MSE or CE
        """

        self.layers = layers
        self.input_size = input_size
        self.error_function = error_map[error_function]

        next_size = self.input_size

        for layer in self.layers:
            next_size = layer.initialize(next_size, optimizer)



    def fit(self, X, Y, epoch, mini_batch_size, learning_rate):
        """
        Update network weights by forward and backward passes
        :param mini_batch_size:
        :param X: input
        :param Y: labels
        :param epoch: trials
        :param learning_rate: learning rate
        :return: a list of losses at each epoch
        """


        sample_size = X.shape[0]
        iter_per_epoch = int(sample_size / mini_batch_size)
        loss_list = []

        for i in range(epoch):

            start = 0
            end = mini_batch_size


            p = np.random.permutation(X.shape[0])
            shuffledX = X[p]
            shuffledY = Y[p]


            loss_sum = 0


            for it in range(iter_per_epoch):

                batchX = shuffledX[start:end]
                batchY = shuffledY[start:end]

                pred, loss = self.__fit_instance(batchX, batchY, learning_rate)

                loss_sum += loss

                start = end
                end += mini_batch_size


            loss_list.append(loss_sum/iter_per_epoch)

        return pred, loss_list


    def predict(self, X):
        """
        Predict for given input using network weights
        :param X: input
        :return: the prediction
        """

        next_X = X

        for layer in self.layers:
            next_X = layer.predict(next_X)

        return next_X


    def __fit_instance(self, X, Y, learning_rate, config=None):

        prediction = self.__call_forward(X)
        loss, residual = self.__calculate_error(prediction, Y)
        self.__call_backward(residual)
        self.__call_update(learning_rate, config=config)

        return prediction, loss


    def __call_forward(self, X):

        next_X = X

        for layer in self.layers:
            next_X = layer.forward_pass(next_X)
        return next_X



    def __call_backward(self, residual):

        next_residual = residual

        for layer in reversed(self.layers):
            next_residual = layer.backward_pass(next_residual)


    def __call_update(self, learning_rate, config):

        for layer in self.layers:
            layer.update(learning_rate, config)



    def __calculate_error(self, pred, actual):
        error, residual = self.error_function(pred, actual)

        return error, residual



