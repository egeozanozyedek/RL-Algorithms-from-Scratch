import numpy as np
from src.NeuralNetwork.Network import Network


class DeepSARSA:

    def __init__(self, state_dim, layers):
        """
        :param state_dim: State vector dimension
        :param layers: Neural Network layers
        """

        self.model = Network(layers, state_dim, "MSE")


    def update(self, state, action, reward, next_state=None, next_action=None, learning_rate=0.1, discount=1, terminate=False):
        """
        The DeepSARSA update rule, updates the weights of the neural network
        :param state: current state
        :param action: current action
        :param reward: reward obtained after committing current action
        :param next_state: state observed after committing current action
        :param next_action: next action chosen on-policy
        :param learning_rate: alpha, the learning rate
        :param discount: gamma, generally 1
        :param terminate: whether the environment is at termination
        """

        update_target = self.q_approx(state)

        if terminate:
            update_target[:, action] = reward
        else:
            update_target[:, action] = reward + discount * self.q_approx(next_state, next_action)

        self.model.fit(state, update_target, epoch=1, learning_rate=learning_rate, momentum_rate=0.1)


    def q_approx(self, state, action=None):
        """
        Approximate Q function, defined by the neural network weights and non-linear functions
        :param state:
        :param action:
        :return:
        """

        if action is None:
            return self.model.predict(state)
        else:
            return self.model.predict(state)[:, action]

