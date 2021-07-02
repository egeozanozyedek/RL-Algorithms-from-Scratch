from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.Train import Train
from matplotlib import pyplot as plt
import gym
import numpy as np
from src.Agents.RandomAgent import RandomAgent




env = gym.make("CartPole-v1").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high

#
# layers = [FullyConnected(8, "sigmoid"),
#           FullyConnected(action_dim, "silu")]
#
# deepsarsa_config = {"state_dim": state_dim, "layers": layers}

rbf_order = 6
basis_function = RBF(rbf_order, state_dim, state_low, state_high)
sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}


trainer = Train(env, "Sarsa", sarsa_config)

with open(f"../weights/Sarsa/goodW.npy", 'rb') as f:
    trainer.model.W = np.load(f)


trainer.play_sarsa(100, 500, True)