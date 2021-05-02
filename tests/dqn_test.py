import random

import numpy as np
import gym
import matplotlib.pyplot as plt
import random as rd

from src.NeuralNetwork.Layers import FullyConnected
from src.NeuralNetwork.Network import Network

env = gym.make('CartPole-v1')

n_a = env.action_space.n
n_s = env.observation_space.shape[0]  # number of features of the states
rd.seed()

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# the discount factor
discount = 0.995

# The learning rate
alpha = 0.01

# The exploration rate
epsilon = 0.3

# The number of episodes used to evaluate the quality of a policy
n_episodes = 5000

# maximum number of steps of a trajectory
max_steps = 499


# Number of time steps, used for updating target network
target_update_count = 0

# Network
layers = [FullyConnected(24, "relu"), FullyConnected(12, "relu"), FullyConnected(n_a, "relu")]
net = Network(layers, n_s, "MSE", optimizer="adam") # Huber loss, He init

# pred, loss = net.fit(X, Y, epoch=200, learning_rate=.4, momentum_rate=.3)
# test_p = net.predict(X_t)

# Target Network
target_net = Network(layers, n_s, "MSE") # Huber loss, He init

# For replay memory
batch_size = 64
gamma = 0.95

replay_memory = []

for i_episode in range(n_episodes):
    print("Episode n° {}".format(i_episode + 1))

    observation = env.reset()
    action = env.action_space.sample() # Initial action

    total_reward = 0


    done = False


    while not done:
        env.render(mode='rgb_array')

        target_update_count += 1


        t = rd.random()
        if t < epsilon: # Exploration
            action = env.action_space.sample()

        else: # Exploitation
            q_vals = net.predict(observation)
            action = np.argmax(q_vals)

        # Observation at t
        observationPrime, reward, done, info = env.step(action)

        # Add to replay memory
        replay_memory.append([observation, action, reward, observationPrime, done])  # s_t, a_t, r_t, s_(t+1)'

        # Update obs
        observation = observationPrime
        total_reward += reward

        # Update the network when 4 steps pass from replay memory(unfreeze)
        if target_update_count % 4 == 0 and target_update_count > 65:

            # Randomly sample replay memory in mini batches
            mini_batch = random.sample(replay_memory, batch_size)
            print(mini_batch)


            # Predict from net
            obs = np.array([sample[0] for sample in mini_batch])
            print(obs, obs.shape)

            # Predict from target, maximize prediction from target
            obs_target = np.array([sample[3] for sample in mini_batch])
            print("obs_target", obs_target, obs_target.shape)
            a = target_net.predict(obs_target)
            print(a, a.shape)
            q_target = np.max(a, axis = 1) # axis?
            print(q_target, q_target.shape)

            # Fit to net
            rewards = np.array([sample[2] for sample in mini_batch])
            Y = (gamma * q_target + rewards).reshape(-1,1)
            Y = np.hstack([Y,Y])
            pred, loss = net.fit(obs, Y, epoch=1, mini_batch_size=64, learning_rate=.001)


        if done:
            print('Total reward at episode {}: {} '.format(i_episode, total_reward))

            # Update the target network
            if target_update_count % 100 == 0:
                target_net.layers = net.layers  # Ege deepcopy?

            print("Episode {} finished after {} timesteps".format(i_episode + 1, target_update_count + 1))
            break


env.close()