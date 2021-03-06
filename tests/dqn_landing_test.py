import pickle

from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.NeuralNetwork.Layers import Dropout
from src.NeuralNetwork.Network import Network
from src.Train import Train
from matplotlib import pyplot as plt
import gym
import numpy as np
from copy import deepcopy
from utils.calc import window_avg
import atari_py
print(atari_py.list_games())

env = gym.make("LunarLander-v2")
#env = gym.make("CartPole-v0").env


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high


# Network
layers = [FullyConnected(150, "relu"), FullyConnected(120, "relu"), FullyConnected(action_dim, "linear")]
net = Network(layers, state_dim, "MSE", optimizer="adam") # Huber loss, He init

# Target Network
target_net = deepcopy(net)


# rbf_order = 6
# basis_function = RBF(rbf_order, state_dim, state_low, state_high)
# sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}

dqn_config = {"state_dim":state_dim, "action_dim":action_dim, "layers":layers, "N": 1000000}


trainer = Train(env, "DQN", dqn_config)

re = []
se = []
le = []

for iter in range(1):
    reward_per_episode, steps_per_episode, loss_per_episode = trainer.train_dqn(episodes=500, time_steps=1000, C=100, min_replay_count=int(64),
                                                              batch_size=int(64), epsilon_decay=0.95, learning_rate=0.001, discount=0.99, epsilon=1, duration = 100000, lr_low=0.0001,
                                                              max_steps=None, decay=True, render=True)
    re.append(reward_per_episode)
    se.append(steps_per_episode)
    le.append(loss_per_episode)

    if iter == 0:
        with open('dqn_lander.pkl', 'wb') as output:
            trainer.model.replay_memory = None
            pickle.dump(trainer.model, output, pickle.HIGHEST_PROTOCOL)


#
np.save( "dqn_lander_reward", np.array(re))
np.save("dqn_lander_steps", np.array(se))
np.save("dqn_lander_loss", np.array(le))

re = np.asarray(re)
se = np.asarray(se)
le = np.asarray(le)

re = np.mean(re, axis=0)
se = np.mean(se, axis=0)
le = np.mean(le, axis=0)


plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(se, "C3")
plt.show()
#plt.savefig("sea_dqnstep_dropout.png")

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(re, "C4")
plt.show()
#plt.savefig("sea_dqn_dropout.png")

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(le, "C5")
plt.show()
#plt.savefig("sea_loss_dropout.png")

# trainer.model.save_weights()

# # Calculate the window average, to understand when it completes the task
# windowed = window_avg(np.array(re), window_size=100, threshold=195)
# print(windowed)
#
# plt.figure(figsize=(12, 8), dpi=160)
# plt.xlabel("Episode")
# plt.ylabel("Time Steps Taken")
# plt.plot(windowed[0], "C4")
# plt.show()
# plt.savefig("atari_dqn_windowed.png")



