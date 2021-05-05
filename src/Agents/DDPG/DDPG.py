import random

from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
from src.Agents.DDPG.Actor import Actor
from src.Agents.DDPG.Critic import Critic
import numpy as np
import copy
from collections import deque



class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)







class DDPG(object):


    def __init__(self, state_dim, action_dim, actor_layers, critic_layers, state_layers, action_layers, env):


        self.actor = Actor(actor_layers, state_dim, "sgd", env.action_space.high[0])
        self.critic = Critic(critic_layers, state_layers, action_layers, state_dim, action_dim, "sgd")
        self.env = env
        self.buffer_size = 50000
        self.batch_size = 64
        self.replay_buffer = deque()
        self.noise = lambda: 0 #OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))



    def train(self, episodes, actor_learning_rate, critic_learning_rate, discount_rate, max_steps, render=False):

        reward_per_episode = []

        for ep in range(episodes):
            # get initial state and action (via policy)
            state = self.env.reset()

            i = 1
            rewards_sum = 0



            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()

                action = self.actor.predict(state.flatten())

                next_state, reward, terminate, info = self.env.step(action)  # commit to action

                self.add_replay((state.flatten(), action.flatten(), np.asarray(reward), next_state.flatten()))

                i += 1
                rewards_sum += reward
                if terminate or i > max_steps:
                    break

                if len(self.replay_buffer) > self.batch_size:

                    replay_batch = self.get_replay_batch()

                    next_action = self.actor.target_predict(replay_batch[-1])
                    self.critic.fit(next_action, replay_batch, critic_learning_rate, discount_rate)


                    grad = self.critic.action_grad(self.batch_size)
                    self.actor.update(grad, replay_batch, actor_learning_rate)




                state = next_state








            reward_per_episode.append(rewards_sum)
            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")


        return reward_per_episode



    def add_replay(self, transition):

        self.replay_buffer.append(transition)

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()



    def get_replay_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, sn = zip(*batch)
        out = np.asarray(s), np.asarray(a), np.asarray(r), np.asarray(sn)
        # print(out[0].shape, out[1].shape)
        return out

