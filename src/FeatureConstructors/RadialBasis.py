import numpy as np
import matplotlib.pyplot as plt


class RBF:

    def __init__(self, order, state_dim, low, high):

        self.order = order
        self.state_dim = state_dim
        self.low = low.astype('float64')
        self.high = high.astype('float64')
        self.centers = None
        if order == 1:
            self.variance = 1
        else:
            self.variance = 2 / (order - 1)
        self.create_centers()


    def create_centers(self):

        partition = np.linspace(0, 1, self.order)

        grid = np.meshgrid(*([partition]*self.state_dim))
        centers = np.vstack(map(np.ravel, grid)).T

        self.centers = centers.tolist()

    def transform(self, state):

        state = self.normalize(state)

        features = np.empty(self.order ** self.state_dim)  # todo: 1d vector, possible problem creator

        for index, c in enumerate(self.centers):
            norm = np.linalg.norm(state - c)**2
            features[index] = np.exp(-norm/(2 * self.variance))

        return features



    def normalize(self, state):
        # return np.exp(state) / (1 + np.exp(state))
        return (state - self.low)/(self.high - self.low)
