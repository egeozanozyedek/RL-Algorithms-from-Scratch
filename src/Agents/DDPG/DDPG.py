from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
import numpy as np
import copy

class DDPG:


    def __init__(self, actor_layers):

        self.actor = Network(actor_layers, X.shape[1], "MSE", optimizer="adam")
        self.critic = None
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)