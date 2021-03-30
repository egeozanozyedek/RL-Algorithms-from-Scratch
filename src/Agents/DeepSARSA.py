import numpy as np
from src.NeuralNetwork.Network import Network


class DeepSARSA:

    def __init__(self, state_dim, layers):

        self.model = Network(layers, state_dim, "MSE")


    def update(self, state, action, reward, next_state=None, next_action=None, learning_rate=0.1, discount=1, terminate=False):

        update_target = self.q_approx(state)

        if terminate:
            update_target[:, action] = reward
        else:
            update_target[:, action] = reward + discount * self.q_approx(next_state, next_action)

        self.model.fit(state, update_target, epoch=1, learning_rate=learning_rate, momentum_rate=0.1)

    def q_approx(self, state, action=None):

        if action is None:
            return self.model.predict(state)
        else:
            return self.model.predict(state)[:, action]

