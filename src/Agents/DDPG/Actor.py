from src.NeuralNetwork.Network import Network
import copy


class Actor(Network):

    def __init__(self, layers, input_size, optimizer, action_bound):
        # print(input_size)
        super().__init__(layers, input_size, "MSE", optimizer=optimizer)
        self.target_layers = copy.deepcopy(self.layers)
        self.action_bound = action_bound



    def update(self, grad, state, learning_rate):

        super()._call_backward(self.action_bound * grad) # todo: maybe minus grad
        self._call_update(learning_rate)


    def predict(self, X):
        return super()._call_forward(X) * self.action_bound



    def target_predict(self, X):

        next_X = X

        for layer in self.target_layers:
            next_X = layer.predict(next_X)

        return next_X * self.action_bound

    def target_update(self, tau):
        for target_layer, layer in zip(self.target_layers, self.layers):
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias
