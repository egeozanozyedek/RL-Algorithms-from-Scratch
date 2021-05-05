from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.NeuralNetwork.Network import Network
from src.Train import Train
from matplotlib import pyplot as plt
import gym
import numpy as np
from copy import deepcopy

env = gym.make("Breakout-ram-v0").env


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high


# Network
layers = [FullyConnected(24, "relu"), FullyConnected(12, "relu"), FullyConnected(action_dim, "linear")]
net = Network(layers, state_dim, "MSE", optimizer="sgd") # Huber loss, He init

# Target Network
target_net = deepcopy(net)


# rbf_order = 6
# basis_function = RBF(rbf_order, state_dim, state_low, state_high)
# sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}

dqn_config = {"state_dim":state_dim, "action_dim":action_dim, "layers":layers, "N": 50000}


trainer = Train(env, "DQN", dqn_config)

re = []
se = []

for i in range(1):
    reward_per_episode, steps_per_episode = trainer.train_dqn(episodes=200, time_steps=1000, C=4, min_replay_count=1024,
                                                              batch_size=128, learning_rate=0.00125, discount=0.7, epsilon=0.01,
                                                              max_steps=None, decay=True, render=True)
    re.append(reward_per_episode)
    se.append(steps_per_episode)

#
re = np.asarray(re)
se = np.asarray(se)

re = np.mean(re, axis=1)
se = np.mean(se, axis=1)

print(steps_per_episode)
print(reward_per_episode)

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(steps_per_episode, "C3")
plt.plot(reward_per_episode, "C4")
plt.show()
plt.savefig("dqnstep.png")

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(reward_per_episode, "C4")
plt.show()
plt.savefig("dqn.png")

# trainer.model.save_weights()
