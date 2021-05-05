from src.NeuralNetwork.Network import Network
import copy


class Actor(Network):

    def __init__(self, layers, input_size, optimizer, action_bound):
        # print(input_size)
        super().__init__(layers, input_size, "MSE", optimizer=optimizer)
        self.target_layers = copy.deepcopy(self.layers)
        self.action_bound = action_bound

        for tl, l in zip(self.target_layers, self.layers):
            assert (tl.weight == l.weight).all()
            assert (tl.bias == l.bias).all()



    def update(self, grad, replay_batch, learning_rate):

        state = replay_batch[0]
        super()._call_forward(state)
        super()._call_backward(grad) # todo: maybe grad @ act instead of *
        self._call_update(learning_rate)


    def predict(self, X):
        res = super().predict(X) * self.action_bound
        return res



    def target_predict(self, X):

        next_X = X

        for layer in self.target_layers:
            next_X = layer.predict(next_X)

        return next_X



    def _call_update(self, learning_rate, config=None):

        tau = 0.001

        for target_layer, layer in zip(self.target_layers, self.layers):
            layer.update(learning_rate, config)
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias

