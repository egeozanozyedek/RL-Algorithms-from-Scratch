import random

from src.NeuralNetwork.Network import Network
from src.NeuralNetwork.Layers import FullyConnected
from src.Agents.DDPG.Actor import Actor
from src.Agents.DDPG.Critic import Critic
import numpy as np
import copy
from collections import deque


class DDPG(object):


    def __init__(self, state_dim, action_dim, actor_layers, critic_layers, state_layers, action_layers, env):

        np.random.seed(11111)
        env.seed(11111)
        self.actor = Actor(actor_layers, state_dim, "adam", env.action_space.high[0])
        self.critic = Critic(critic_layers, state_layers, action_layers, state_dim, action_dim, "adam")
        self.env = env
        self.buffer_size = 100000
        self.batch_size = 64
        self.replay_buffer = deque()
        self.noise = lambda: 0 #OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))



    def train(self, episodes, actor_learning_rate, critic_learning_rate, discount_rate, max_steps, render=False):

        reward_per_episode = []
        Q_loss = []
        policy_loss = []

        for ep in range(episodes):
            # get initial state and action (via policy)
            state = self.env.reset()
            i = 1
            rewards_sum = 0
            Q_loss_sum = 0
            policy_loss_sum = 0

            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()

                action = self.actor.predict(state.T) + 1. / (1 + ep * i)

                next_state, reward, terminate, info = self.env.step(action)  # commit to action

                self.add_replay((state.flatten().astype("float32"), action.flatten().astype("float32"),
                                 np.asarray(reward).astype("float32"), next_state.flatten().astype("float32"), np.asarray(terminate)))



                if len(self.replay_buffer) > self.batch_size:

                    replay_batch = self.get_replay_batch()

                    next_action = self.actor.target_predict(replay_batch[-2])
                    critic_loss = self.critic.fit(next_action, replay_batch, critic_learning_rate, discount_rate)


                    grad, actor_loss = self.critic.action_grad(replay_batch[0], self.actor.predict(replay_batch[0]), self.batch_size)
                    self.actor.update(grad, replay_batch[0], actor_learning_rate)


                    self.critic.target_update(0.001)
                    self.actor.target_update(0.001)

                    Q_loss_sum += critic_loss
                    policy_loss_sum += actor_loss


                i += 1
                rewards_sum += reward
                if terminate or i > max_steps:
                    break

                state = next_state

            Q_loss.append(Q_loss_sum/i)
            policy_loss.append(policy_loss_sum/i)
            reward_per_episode.append(rewards_sum)
            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")


        return reward_per_episode



    def add_replay(self, transition):

        self.replay_buffer.append(transition)

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()



    def get_replay_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, sn, d = zip(*batch)
        out = np.asarray(s), np.asarray(a), np.asarray(r).reshape(-1, 1), np.asarray(sn), np.asarray(d).reshape(-1, 1)

        return out

