import numpy as np
from src.NeuralNetwork.Layers.nn_toolkit import activation_map
from numba import jit


class FullyConnected(object):


    def __init__(self, nodes, activation):
        """
        Initializes important variables
        self.opv: private member of class, opv being an abbreviation for one pass variables

        :param nodes: output of layer
        :param activation: the activation function of layer
        """


        self.weight = None
        self.bias = None

        self.opv = {"X": None, "dW": None, "db": None, "dA": None, "mW": 0, "mb": 0, "sW": 0, "sb": 0}

        self.nodes = nodes
        self.activation = activation_map[activation]
        self.optimizer = None
        self.iteration = 1



    def initialize(self, input_size, optimizer):
        """
        Xavier Initialization for layer parameters
        :param input_size: input size of this layer
        :return: output size of this layer / layer nodes
        """

        self.optimizer = optimizer
        init = np.sqrt(6/(input_size + self.nodes))
        self.weight = np.random.uniform(-init, init, size=(input_size, self.nodes))
        self.bias = np.zeros((1, self.nodes))


        return self.nodes



    def forward_pass(self, X):
        """
        Forward pass of layer
        :param X:
        :return:
        """
        # print(X.shape, self.weight.shape, self.bias.shape)
        if X.ndim is 1:
            X = X.reshape(1, -1)


        self.opv["X"] = X
        potential = X @ self.weight + self.bias
        phi, self.opv["dA"] = self.activation(potential)

        # print(X.shape, self.weight.shape)

        return phi


    def backward_pass(self, residual):
        """
        Backward pass of layer, gradient descent
        :param residual: The residual error gradient from the earlier layer
        :return: the next iteration residual
        """

        # print(self.nodes, residual.shape, self.opv["dA"].shape )
        # print(residual.shape, self.opv["X"].shape, self.weight.shape, self.opv["dA"])

        residual *= self.opv["dA"]  # updater residual term
        self.opv["dW"] = self.opv["X"].T @ residual
        self.opv["db"] = residual.sum(axis=0, keepdims=True)

        return residual @ self.weight.T




    def update(self, learning_rate, config):
        """

        Updates layer weight and bias

        dW: gradient of weight(t)
        db: gradient of bias(t)

        mW: momentum of weight(t)
        mb: momentum(t) of bias(t)


        :param learning_rate: learning rate for update
        :param momentum_rate: momentum_rate for updatte
        """


        if self.optimizer == "sgd":

            if config is None:
                momentum_rate = 0.1
            else:
                momentum_rate = config

            self.opv["mW"] = learning_rate * self.opv["dW"] + momentum_rate * self.opv["mW"]
            self.opv["mb"] = learning_rate * self.opv["db"] + momentum_rate * self.opv["mb"]

            self.weight -= self.opv["mW"]
            self.bias -= self.opv["mb"]

        if self.optimizer == "adam":

            if config is None:
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
            else:
                beta1, beta2, epsilon = config

            self.opv["mW"] = (1 - beta1) * self.opv["dW"] + beta1 * self.opv["mW"]
            self.opv["mb"] = (1 - beta1) * self.opv["db"] + beta1 * self.opv["mb"]

            self.opv["sW"] = (1 - beta2) * np.square(self.opv["dW"]) + beta2 * self.opv["sW"]
            self.opv["sb"] = (1 - beta2) * np.square(self.opv["db"]) + beta2 * self.opv["sb"]


            # self.opv["mW"] /= 1 - beta1 ** self.iteration
            # self.opv["mb"] /= 1 - beta1 ** self.iteration
            #
            # self.opv["sW"] /= 1 - beta2 ** self.iteration
            # self.opv["sb"] /= 1 - beta2 ** self.iteration


            self.weight -= learning_rate * self.opv["mW"]/(np.sqrt(self.opv["sW"]) + epsilon)
            self.bias -= learning_rate * self.opv["mb"]/(np.sqrt(self.opv["sb"]) + epsilon)

            self.iteration += 1



    def predict(self, X):
        """
        Predicts given output using layer weight and bias
        :param X: Input vector/matrix
        :return: the predicted value
        """

        potential = X @ self.weight + self.bias
        return self.activation(potential)[0]


