import numpy as np
from collections import deque

class DQN:

    def __init__(self, state_dim, action_dim, layers, batch_size, N):
        """

        :param state_dim:
        :param action_dim:
        :param layers:
        :param batch_size: Batch size for replay memory.
        :param N: Experience replay (pool of stored samples) capacity size (max)
        """

        self.network =
        self.target_network =


        self.replay_memory  = deque(maxlen=N)



    def update(self, state, action, reward, next_state=None, next_action=None, learning_rate=0.5, discount=0.5, terminate=False):


    def q_approx(self, state, action=None):
        """
        Return action.

        :param state: state vector
        :param action: discrete action
        :return: the approximated q-value
        """


        if action is None:
            return self.network.predict(state)
        else:
            return self.network.predict(state)[:,action] #??


