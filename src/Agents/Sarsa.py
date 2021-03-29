import numpy as np
from numba import jit

# todo: we should implement the tiling thing in the book for continuous action spaces

class Sarsa:

    def __init__(self, state_dim, action_dim, basis_function):

        self.basis_function = basis_function
        self.W = np.zeros((self.basis_function.order ** state_dim, action_dim))  # weight vector, see Ch.10 or 9,



    def update(self, state, action, reward, next_state=None, next_action=None, alpha=0.5, gamma=0.5, terminate=False):


        q_approx_grad = self.transform(state) # wrote it like this so that its more understandable

        if terminate:
            update_target = reward - self.q_approx(state, action)
        else:
            update_target = reward + gamma * self.q_approx(next_state, next_action) - self.q_approx(state, action)

        self.W[:, action] += alpha * update_target * q_approx_grad



    def q_approx(self, state, action=None):
        '''
        approximate q function, see eqn 10.3/link:https://gist.github.com/martinholub/c4860006d0cf3fbe87a79a054a9c98cd
        :param state:
        :param action:
        :return:
        '''

        state = self.transform(state)

        # print(self.W)
        if action is None:
            return self.W.T @ state
        else:
            return self.W[:, action].T @ state




    def transform(self, state):

        if self.basis_function is not None:
            return self.basis_function.transform(state)
        else:
            return state







