from src.FeatureConstructors.RadialBasis import RBF
from src.NeuralNetwork.Layers import FullyConnected
from src.Train import Train
from matplotlib import pyplot as plt
import gym



env = gym.make("Breakout-ram-v0").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high


layers = [FullyConnected(10, "sigmoid"),
          FullyConnected(50, "sigmoid"),
          FullyConnected(10, "sigmoid"),
          FullyConnected(action_dim, "silu")]

deepsarsa_config = {"state_dim": state_dim, "layers": layers}


#
# rbf_order = 6
# basis_function = RBF(rbf_order, state_dim, state_low, state_high)
# sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}







trainer = Train(env, "DeepSARSA", deepsarsa_config)

reward_per_episode, steps_per_episode = trainer.train(episodes=50000, learning_rate=1.5e-2, discount=1, epsilon=0.15, max_steps=None, render=False)


plt.figure(figsize=(24, 8), dpi=160)
plt.subplot(1, 2, 1)
plt.xlabel("Episode")
plt.ylabel("Reward of Episode")
plt.plot(reward_per_episode, "C2")
plt.subplot(1, 2, 2)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(steps_per_episode, "C3")
plt.show()
