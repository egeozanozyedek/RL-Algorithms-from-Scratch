from src.Agents.DDPG.DDPG import DDPG
import gym
from src.NeuralNetwork.Layers.FullyConnected import FullyConnected
import pickle



env = gym.make("BipedalWalker-v3")
print(env.observation_space.shape, env.action_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]



# actor_layers = [FullyConnected(256, "relu"), FullyConnected(256, "relu"), FullyConnected(action_dim, "tanh")]
# critic_layers = [FullyConnected(256, "relu"), FullyConnected(256, "relu"), FullyConnected(1, "relu")]
# state_layers = [FullyConnected(16, "relu"), FullyConnected(32, "relu")]
# action_layers = [FullyConnected(32, "relu")]

actor_layers = [FullyConnected(600, "relu"), FullyConnected(300, "relu"), FullyConnected(600, "relu"), FullyConnected(action_dim, "tanh")]
critic_layers = [FullyConnected(300, "relu"), FullyConnected(600, "relu"), FullyConnected(1, "linear")]
state_layers = [FullyConnected(600, "relu"), FullyConnected(300, "linear")]
action_layers = [FullyConnected(300, "linear")]


# actor_layers = [FullyConnected(320, "relu"), FullyConnected(640, "relu"), FullyConnected(action_dim, "tanh")]
# critic_layers = [FullyConnected(640, "relu"), FullyConnected(1, "linear")]
# state_layers = [FullyConnected(320, "relu"), FullyConnected(640, "linear")]
# action_layers = [FullyConnected(640, "linear")]



# actor_layers = [FullyConnected(64, "relu"),  FullyConnected(action_dim, "linear")]
# critic_layers = [FullyConnected(64, "relu"), FullyConnected(1, "relu")]
# state_layers = [FullyConnected(32, "relu")]
# action_layers = [FullyConnected(32, "relu")]


ddpg = DDPG(state_dim, action_dim, actor_layers, critic_layers, state_layers, action_layers, env)

episodes = 2000
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
tau = 0.001
discount_rate = 0.99
max_steps = 700

reward_per_episode, Q_loss, policy_loss = ddpg.train(episodes, actor_learning_rate, critic_learning_rate, discount_rate, tau, max_steps, render=True)

with open("agent47", 'wb') as f:
    pickle.dump((ddpg, reward_per_episode, Q_loss, policy_loss), f)