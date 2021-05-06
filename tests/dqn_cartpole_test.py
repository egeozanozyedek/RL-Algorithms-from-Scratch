import pickle

from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.NeuralNetwork.Network import Network
from src.Train import Train
from matplotlib import pyplot as plt
import gym
import numpy as np
from copy import deepcopy
from utils.calc import window_avg
import atari_py




env = gym.make("CartPole-v0").env
env.seed(8)
env.observation_space.seed(8)
env.action_space.seed(8)
np.random.seed(7)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high


# Network
layers = [FullyConnected(12, "relu"), FullyConnected(24, "relu"), FullyConnected(action_dim, "linear")]
net = Network(layers, state_dim, "MSE", optimizer="adam") # Huber loss, He init

# Target Network
target_net = deepcopy(net)


# rbf_order = 6
# basis_function = RBF(rbf_order, state_dim, state_low, state_high)
# sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}

dqn_config = {"state_dim":state_dim, "action_dim":action_dim, "layers":layers, "N": 80000}


trainer = Train(env, "DQN", dqn_config)

re = []
se = []
le = []

for iter in range(5):
    print("iter", iter)

    if iter == 2 or iter == 1:
        render = False
    else:
        render = False
    reward_per_episode, steps_per_episode, loss_per_episode = trainer.train_dqn(episodes=400, time_steps=200, C=4, min_replay_count=int(32),
                                                              batch_size=int(32), learning_rate=0.002, discount=0.95, epsilon=1, duration = 30, lr_low=0.0005,
                                                              max_steps=None, decay=True, render=render)


    if iter == 4:
        with open('dqn_cartpole_save.pkl', 'wb') as output:
            trainer.model.replay_memory = None
            pickle.dump(trainer.model, output, pickle.HIGHEST_PROTOCOL)

    print(np.array(reward_per_episode).mean())
    re.append(reward_per_episode)
    se.append(steps_per_episode)
    le.append(loss_per_episode)

#

np.save( "dqn_cartpole_rewards", np.array(re))
np.save("dqn_cartpole_steps", np.array(se))
np.save("dqn_cartpole_loss", np.array(le))

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
plt.savefig("dqnstep.png")

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(re, "C4")
plt.show()
plt.savefig("dqn.png")

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(le, "C5")
plt.show()
plt.savefig("loss.png")

# trainer.model.save_weights()

# Calculate the window average, to understand when it completes the task
windowed = window_avg(np.array(re), window_size=100, threshold=195)
print(windowed)

plt.figure(figsize=(12, 8), dpi=160)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(windowed[0], "C4")
plt.show()
plt.savefig("dqn_windowed.png")



