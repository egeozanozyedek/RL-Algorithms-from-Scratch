import numpy as np
import pickle

# todo: we should implement the tiling thing in the book for continuous action spaces

class Sarsa:

    def __init__(self, state_dim, action_dim, basis_function):
        """
        :param state_dim: Dimension of state vector
        :param action_dim: The number of actions available in environment
        :param basis_function: An object, RBF
        """

        self.basis_function = basis_function
        self.W = np.zeros((self.basis_function.order ** state_dim, action_dim))  # weight vector, see Ch.10 or 9,



    def update(self, state, action, reward, next_state=None, next_action=None, learning_rate=0.5, discount=0.5, terminate=False):
        """
        The SARSA update rule, updates the weight of corresponding action using S, A, R, S', A'
        :param state: current state
        :param action: current action
        :param reward: reward obtained after committing current action
        :param next_state: state observed after committing current action
        :param next_action: next action chosen on-policy
        :param learning_rate: alpha, the learning rate
        :param discount: gamma, generally 1
        :param terminate: whether the environment is at termination
        """


        q_approx_grad = self.transform(state)  # wrote it like this so that its more understandable

        if terminate:
            update_target = reward - self.q_approx(state, action)
        else:
            update_target = reward + discount * self.q_approx(next_state, next_action) - self.q_approx(state, action)

        self.W[:, action] += learning_rate * update_target * q_approx_grad



    def q_approx(self, state, action=None):
        """
        Approximate Q function, defined by the linear equation W.T @ x
        :param state: state vector
        :param action: discrete action
        :return: the approximated q-value
        """

        x = self.transform(state)

        # print(self.W)
        if action is None:
            return self.W.T @ x
        else:
            return self.W[:, action].T @ x


    def transform(self, state):
        """
        Constructs features following the given basis function (RBF for now)
        :param state: the state vector
        :return: the constructed features
        """

        return self.basis_function.transform(state)




    def save_weights(self, dir="../weights/Sarsa"):
        """
        A function to save weights of SARSA
        :param dir: dierctory
        """
        np.save(f"{dir}/W", self.W)








