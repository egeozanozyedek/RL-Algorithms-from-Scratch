from src.NeuralNetwork.Network import Network
import copy


class Critic(Network):

    def __init__(self, layers, input_size, optimizer):

        action_net =
        state_net =
        merge_net =

        self.critic = Network(layers, input_size, "mse", optimizer=)
        self.target = copy.deepcopy(self.critic)




    def fit(self, X, Y, epoch, mini_batch_size, learning_rate):
        return self.critic.fit(X, Y, epoch, mini_batch_size, learning_rate)



    def target_update(self, tau):

        for target_layer, layer in zip(self.target.layers, self.critic.layers):
            target_layer.weight = (1 - tau) * target_layer.weight + tau * layer.weight
            target_layer.bias = (1 - tau) * target_layer.bias + tau * layer.bias



    def target_predict(self, X):
        return self.target.predict(X)


    def critic_predict(self, X):
        for net in self.critic:
        return self.critic.predict(X)