import numpy as np
from src.NeuralNetwork.Layers.nn_toolkit import activation_map




class FullyConnected(object):


    def __init__(self, nodes, activation):
        """
        Initializes important variables
        self.__opv: private member of class, opv being an abbreviation for one pass variables

        :param nodes: output of layer
        :param activation: the activation function of layer
        """


        self.weight = None
        self.bias = None

        self.__opv = {"X": None, "dW": None, "db": None, "dA": None, "mW": 0, "mb": 0}

        self.nodes = nodes
        self.activation = activation_map[activation]


    def initialize(self, input_size):
        """
        Xavier Initialization for layer parameters
        :param input_size: input size of this layer
        :return: output size of this layer / layer nodes
        """

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

        if X.ndim is 1:
            X = X.reshape(1, -1)

        self.__opv["X"] = X
        potential = X @ self.weight + self.bias
        phi, self.__opv["dA"] = self.activation(potential)

        return phi


    def backward_pass(self, residual):
        """
        Backward pass of layer, gradient descent
        :param residual: The residual error gradient from the earlier layer
        :return: the next iteration residual
        """

        # print(self.nodes, residual.shape, self.__opv["dA"].shape )

        residual *= self.__opv["dA"]  # updater residual term
        self.__opv["dW"] = self.__opv["X"].T @ residual
        self.__opv["db"] = residual.sum(axis=0, keepdims=True)

        return residual @ self.weight.T




    def update(self, learning_rate, momentum_rate):
        """

        Updates layer weight and bias

        dW: gradient of weight(t)
        db: gradient of bias(t)

        mW: momentum of weight(t)
        mb: momentum(t) of bias(t)


        :param learning_rate: learning rate for update
        :param momentum_rate: momentum_rate for updatte
        """


        self.__opv["mW"] = learning_rate * self.__opv["dW"] + momentum_rate * self.__opv["mW"]
        self.__opv["mb"] = learning_rate * self.__opv["db"] + momentum_rate * self.__opv["mb"]

        self.weight -= self.__opv["mW"]
        self.bias -= self.__opv["mb"]



    def predict(self, X):
        """
        Predicts given output using layer weight and bias
        :param X: Input vector/matrix
        :return: the predicted value
        """

        potential = X @ self.weight + self.bias
        return self.activation(potential)[0]


