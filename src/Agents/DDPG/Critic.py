from src.NeuralNetwork.Network import Network
import copy
import numpy as np


class Critic(Network):

    def __init__(self, layers, state_layers, action_layers, state_dim, action_dim, optimizer):



        self.state_layers = state_layers
        state_next_size = state_dim
        for state_layer in self.state_layers:
            state_next_size = state_layer.initialize(state_next_size, optimizer)

        self.action_layers = action_layers
        action_next_size = action_dim
        for action_layer in self.action_layers:
            action_next_size = action_layer.initialize(action_next_size, optimizer)

        self.state_dim = state_next_size
        self.action_dim = action_next_size

        super().__init__(layers, action_next_size + state_next_size , "MSE", optimizer=optimizer)


        self.target_layers = copy.deepcopy(self.layers)
        self.target_state_layers = copy.deepcopy(self.state_layers)
        self.target_action_layers = copy.deepcopy(self.action_layers)


    # def fit(self, next_action, replay_batch, learning_rate, discount_rate, tau=0.01):
    #
    #     state, action, reward, next_state, done = replay_batch
    #     reward = np.reshape(reward, (-1, 1))
    #     target_Q = self.target_predict((next_state, next_action))
    #     Y = np.empty_like(action)
    #
    #     for i in range(len(Y)):
    #         if not done[i]:
    #             Y[i] = reward[i] + discount_rate * target_Q[i]
    #         else:
    #             Y[i] = reward[i]
    #
    #     print("Y:", Y.shape, Y.min(), Y.max())


    def fit(self, state, action, Y, learning_rate, a=None):
        # print(state.shape, action.shape, Y.shape)
        _, loss = super()._fit_instance((state, action), Y, learning_rate)

        return loss




    def predict(self, X):
        ns, na = X

        for state_layer in self.state_layers:
            ns = state_layer.predict(ns)

        for action_layer in self.action_layers:
            na = action_layer.predict(na)

        concat = np.hstack((ns, na))
        # concat = ns + na

        next_X = concat

        for layer in self.layers:
            next_X = layer.predict(next_X)

        return next_X


    def target_predict(self, X):
        ns, na = X

        for state_layer in self.target_state_layers:
            ns = state_layer.predict(ns)

        for action_layer in self.target_action_layers:
            na = action_layer.predict(na)
        #
        concat = np.hstack((ns, na))
        # concat = ns + na

        next_X = concat

        for layer in self.target_layers:
            next_X = layer.predict(next_X)

        return next_X





    def action_grad(self, state, action, N):


        out = self._call_forward((state, action))
        loss = out.sum()

        next_residual = np.ones_like(out)/N

        for layer in reversed(self.layers):
            next_residual = layer.backward_pass(next_residual)

        action_residual = next_residual[:, self.state_dim:]
        # action_residual = next_residual

        for action_layer in reversed(self.action_layers):
            action_residual = action_layer.backward_pass(action_residual)


        return -action_residual, out, loss



    def _call_forward(self, X):

        ns, na = X

        for state_layer in self.state_layers:
            ns = state_layer.forward_pass(ns)

        for action_layer in self.action_layers:
            na = action_layer.forward_pass(na)


        concat = np.hstack((ns, na))
        # concat = ns + na


        next_X = concat
        for layer in self.layers:
            next_X = layer.forward_pass(next_X)


        return next_X





    def _call_backward(self, residual):

        next_residual = residual
        # print(residual.shape)

        for layer in reversed(self.layers):
            next_residual = layer.backward_pass(next_residual)


        state_residual, action_residual = next_residual[:, :self.state_dim], next_residual[:, self.state_dim:]
        # state_residual = next_residual
        # action_residual = next_residual


        for state_layer in reversed(self.state_layers):
            state_residual = state_layer.backward_pass(state_residual)

        for action_layer in reversed(self.action_layers):
            action_residual = action_layer.backward_pass(action_residual)




    def _call_update(self, learning_rate, config=None):

        for layer in self.layers:
            layer.update(learning_rate, config)

        for layer in self.state_layers:
            layer.update(learning_rate, config)

        for layer in self.action_layers:
            layer.update(learning_rate, config)


    def target_update(self, tau):

        for target_layer, layer in zip(self.target_layers, self.layers):
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias

        for target_layer, layer in zip(self.target_state_layers, self.state_layers):
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias

        for target_layer, layer in zip(self.target_action_layers, self.action_layers):
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias

