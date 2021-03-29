import gym
import numpy as np
import matplotlib.pyplot as plt

# Initializations
env = gym.make('MountainCar-v0')

num_actions = env.action_space.n
dim = env.observation_space.high.size

# Parameters
# one set which converges in around 1200 episodes
# 4 rows, 4 cols, eps = 0.1, Lambda = 0.5, alpha = 0.008, gamma = 0.99
num_rbf = 4 * np.ones(num_actions).astype(int)
width = 1. / (num_rbf - 1.)
rbf_sigma = width[0] / 2.
epsilon = 0.1
epsilon_final = 0.1
Lambda = 0.5
alpha = 0.01
gamma = 0.99
num_episodes = 2000
num_timesteps = 200

xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high
num_ind = np.prod(num_rbf)
activations = np.zeros(num_ind)
new_activations = np.zeros(num_ind)
theta = np.zeros((num_ind, num_actions))
rbf_den = 2 * rbf_sigma ** 2
epsilon_coefficient = (epsilon - epsilon_final) ** (1. / num_episodes)
ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)