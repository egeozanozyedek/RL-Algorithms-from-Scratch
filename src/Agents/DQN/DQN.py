import numpy as np
from collections import deque
from copy import deepcopy
from src.NeuralNetwork.Network import Network


class DQN:

    def __init__(self, state_dim, action_dim, layers, N):
        """

        :param state_dim:
        :param action_dim:
        :param layers:
        :param batch_size: Batch size for replay memory.
        :param N: Experience replay (pool of stored samples) capacity size (max)
        """

        self.network = Network(layers, state_dim, "MSE", optimizer="adam") # Huber loss, He init
        self.target_network = deepcopy(self.network)
        self.replay_memory = deque(maxlen=N)


    def update(self, mini_batch, learning_rate=0.1, discount=1,
               terminate=False):
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
        # print("mini-batch")
        # print(mini_batch)

        states = np.array([transition[0] for transition in mini_batch])
        next_states = np.array([transition[3] for transition in mini_batch]) # actionsize

        q_values = self.q_approx(states)
        next_q_values = self.q_approx(next_states, target=True)

        #rate = 0.7
        rate = 0.9
        # Find y_i
        X = []
        Y = []
        for index, (state, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount * np.max(next_q_values[index])
            else:
                max_future_q = reward

            # Y.append(max_future_q)
            # X.append(state)

            current_qs = q_values[index]
            current_qs[action] = (1 - rate) * current_qs[action] + rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        x = np.array(X)
        # print(x.shape)
        y = np.array(Y)
        # print(y.shape)
        self.network.fit(x, y, epoch=1, mini_batch_size=len(mini_batch), learning_rate=learning_rate)



    def q_approx(self, state, action=None, target=False):
        """
        Return action.

        :param state: state vector
        :param action: discrete action
        :return: the approximated q-value
        """

        a = self.network.predict(state)


        if target is False:
            if action is None:
                return a
            else:
                return self.network.predict(state)[:,action] #??
        else:
            if action is None:
                return self.target_network.predict(state)
            else:
                return self.target_network.predict(state)[:,action] #??



    def update_target(self):
        self.target_network = deepcopy(self.network)