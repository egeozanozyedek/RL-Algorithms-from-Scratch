import numpy as np


class RBF:

    def __init__(self, order, state_dim, low, high):
        """
        Radial Basis Function for feature construction
        :param order: n divisions for each dimension of the state vector
        :param state_dim: dimension of state vector
        :param low: min values of state features, for normalization
        :param high: max values of state features, for normalization
        """

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
        """
        Creates centers of each RBF, centers are evenly distributed
        """

        partition = np.linspace(0, 1, self.order)

        grid = np.meshgrid(*([partition]*self.state_dim))
        centers = np.vstack(map(np.ravel, grid)).T

        self.centers = centers.tolist()


    def transform(self, state):
        """
        Creates features
        :param state: the state vector from which the features will be created
        :return: constructed features
        """

        state = self.normalize(state)

        features = np.empty(self.order ** self.state_dim)  # todo: 1d vector, possible problem creator

        for index, c in enumerate(self.centers):
            norm = np.linalg.norm(state - c)**2
            features[index] = np.exp(-norm/(2 * self.variance))


        features = np.where(features > 0.95, 1, 0)

        return features



    def normalize(self, state):
        """
        Normalizes given input
        :param state: the state vector from which the features will be created
        :return: normalized state vector
        """
        # return np.exp(state) / (1 + np.exp(state))
        return (state - self.low)/(self.high - self.low)
