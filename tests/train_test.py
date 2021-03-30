from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.Train import Train
from matplotlib import pyplot as plt
import gym
import numpy as np


env = gym.make("CartPole-v1").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high


layers = [FullyConnected(8, "sigmoid"),
          FullyConnected(action_dim, "silu")]

deepsarsa_config = {"state_dim": state_dim, "layers": layers}



# rbf_order = 7
# basis_function = RBF(rbf_order, state_dim, state_low, state_high)
# sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}







trainer = Train(env, "DeepSARSA", deepsarsa_config)

re = []
se = []

for i in range(10):
    reward_per_episode, steps_per_episode = trainer.train(episodes=1000, learning_rate=0.08, discount=1, epsilon=1, max_steps=200, decay=True, render=False)
    re.append(reward_per_episode)
    se.append(steps_per_episode)


re = np.asarray(re)
se = np.asarray(se)

plt.figure(figsize=(24, 8), dpi=160)
plt.subplot(1, 2, 1)
plt.xlabel("Episode")
plt.ylabel("Reward of Episode")
plt.plot(np.mean(re, axis=1), "C2")
plt.subplot(1, 2, 2)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(np.mean(se, axis=1), "C3")
plt.show()
