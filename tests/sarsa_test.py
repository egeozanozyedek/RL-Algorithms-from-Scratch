import numpy as np
import gym
import matplotlib.pyplot as plt


def update(Q, state, action, reward, next_state, next_action, alpha, gamma):
        return Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])


def policy(env, epsilon, Q, state):
    r = np.random.rand()
    if r < epsilon:
        return np.argmax(Q[state])
    else:
        return env.action_space.sample()


def initialize(state_space, action_space):
    Q = np.zeros((state_space, action_space))
    return Q


def train_SARSA(episodes, alpha, gamma, epsilon):
    rewards = []
    env = gym.make("FrozenLake-v0")
    print(env.action_space, env.observation_space)
    Q = initialize(env.observation_space.n, env.action_space.n)

    for ep in range(episodes):

        env.reset()
        state = env.observation_space.sample()
        action = policy(env, epsilon, Q, state)
        done = False
        i = 0
        reward_sum = 0

        while not done:

            # env.render()

            next_state, reward, done, info = env.step(action)
            next_action = policy(env, epsilon, Q, next_state)

            Q[state, action] = update(Q, state, action, reward, next_state, next_action, alpha, gamma)

            state = next_state
            action = next_action

            reward_sum += reward
            i += 1

        rewards.append(reward_sum/i)

    plt.scatter(range(len(rewards)), rewards)
    plt.show()



episodes = 10000
gamma = 0.7
alpha = 0.5
epsilon = 0.85

train_SARSA(episodes, alpha, gamma, epsilon)