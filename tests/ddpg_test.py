from src.Agents.DDPG.DDPG import DDPG
import gym
from src.NeuralNetwork.Layers.FullyConnected import FullyConnected



env = gym.make("Pendulum-v0").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print(state_dim, action_dim)

# actor_layers = [FullyConnected(256, "relu"), FullyConnected(256, "relu"), FullyConnected(action_dim, "tanh")]
# critic_layers = [FullyConnected(256, "relu"), FullyConnected(256, "relu"), FullyConnected(1, "relu")]
# state_layers = [FullyConnected(16, "relu"), FullyConnected(32, "relu")]
# action_layers = [FullyConnected(32, "relu")]


actor_layers = [FullyConnected(300, "relu"), FullyConnected(600, "relu"), FullyConnected(action_dim, "tanh")]
critic_layers = [FullyConnected(600, "relu"), FullyConnected(1, "relu")]
state_layers = [FullyConnected(300, "relu"), FullyConnected(600, "linear")]
action_layers = [FullyConnected(600, "linear")]



# actor_layers = [FullyConnected(64, "relu"),  FullyConnected(action_dim, "linear")]
# critic_layers = [FullyConnected(64, "relu"), FullyConnected(1, "relu")]
# state_layers = [FullyConnected(32, "relu")]
# action_layers = [FullyConnected(32, "relu")]


ddpg = DDPG(state_dim, action_dim, actor_layers, critic_layers, state_layers, action_layers, env)

episodes = 40
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
discount_rate = 0.99
max_steps = 1000

ddpg.train(episodes, actor_learning_rate, critic_learning_rate, discount_rate, max_steps, render=False)