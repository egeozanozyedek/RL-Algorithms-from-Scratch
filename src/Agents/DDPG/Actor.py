from src.NeuralNetwork.Network import Network
import copy


class Actor(Network):

    def __init__(self, layers, input_size, optimizer):

        self.actor = Network(layers, input_size, optimizer=optimizer)
        self.target = copy.deepcopy(self.actor)

