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

        print(env.action_space.high)
        self.actor = Actor(actor_layers, state_dim, "adam", env.action_space.high)
        self.critic = Critic(critic_layers, state_layers, action_layers, state_dim, action_dim, "adam")
        self.env = env
        self.buffer_size = 1000000
        self.batch_size = 100
        self.replay_buffer = deque()
        self.epsilon = 1
        self.action_dim = action_dim



    def train(self, episodes, actor_learning_rate, critic_learning_rate, discount_rate, tau, max_steps, info_tuple, render=False):

        reward_per_episode, Q_loss, policy_loss = info_tuple

        for ep in range(episodes):
            # get initial state and action (via policy)
            state = self.env.reset()
            i = 1
            rewards_sum = 0
            Q_loss_sum = 0
            policy_loss_sum = 0
            if self.epsilon > 0.01:
                self.epsilon *= 0.995
            actor_learning_rate *= 0.99995
            critic_learning_rate *= 0.99995

            while True:  # loop controlled with termination of state, run until done

                if render is True and ep > 500:  # for visualization
                    self.env.render()

                if np.random.random() < self.epsilon:
                    action = np.clip(self.actor.predict(state.T) + np.random.normal(self.action_dim), -1, 1) * self.env.action_space.high
                else:
                    action = self.actor.predict(state.T)


                next_state, reward, terminate, info = self.env.step(action.flatten())  # commit to action

                self.add_replay((state.flatten(), action.flatten(), reward, next_state.flatten(), terminate))

                if len(self.replay_buffer) > self.batch_size:

                    states, actions, rewards, next_states, dones = self.get_replay_batch()

                    next_actions = self.actor.target_predict(next_states)
                    target_Q = self.critic.target_predict((next_states, next_actions))
                    Y = np.empty((self.batch_size, 1))

                    # Y = rewards + discount_rate * target_Q
                    # Y[dones] = rewards[dones]

                    for j in range(len(Y)):
                        if not dones[j]:
                            Y[j] = rewards[j] + discount_rate * target_Q[j]
                        else:
                            Y[j] = rewards[j]

                    # print("Y:", Y.shape, Y.min(), Y.max())


                    critic_loss = self.critic.fit(states, actions, Y, critic_learning_rate)

                    action_prediction = self.actor.predict(states)
                    grad, critic_prediction, actor_loss = self.critic.action_grad(states, action_prediction, self.batch_size)
                    self.actor.update(grad, critic_prediction, actor_learning_rate)

                    self.critic.target_update(tau)
                    self.actor.target_update(tau)

                    Q_loss_sum += critic_loss
                    policy_loss_sum += actor_loss



                # print(f"Step: {i}")
                i += 1
                rewards_sum += reward
                state = next_state

                if terminate or i > max_steps:
                    break



            Q_loss.append(Q_loss_sum/i)
            policy_loss.append(policy_loss_sum/i)
            reward_per_episode.append(rewards_sum)
            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")


        # return reward_per_episode, Q_loss, policy_loss



    def add_replay(self, transition):

        self.replay_buffer.append(transition)

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()



    def get_replay_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        # s, a, r, sn, d = zip(*batch)
        # out = np.asarray(s), np.asarray(a), np.asarray(r).reshape(-1, 1), np.asarray(sn), np.asarray(d).reshape(-1, 1)

        s = np.asarray([sample[0] for sample in batch])
        a = np.asarray([sample[1] for sample in batch])
        r = np.asarray([sample[2] for sample in batch])
        sn = np.asarray([sample[3] for sample in batch])
        d = np.asarray([sample[4] for sample in batch])

        out = s, a, r, sn, d
        return out



    def play(self, episodes, max_steps, render=False):

        reward_per_episode = []

        for ep in range(episodes):
            # get initial state and action (via policy)
            state = self.env.reset()
            i = 1
            rewards_sum = 0


            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()

                action = self.actor.predict(state.T)
                next_state, reward, terminate, info = self.env.step(action.flatten())  # commit to action
                state = next_state
                # print(f"Step: {i}")
                i += 1
                rewards_sum += reward


                if terminate or i > max_steps:
                    break


            reward_per_episode.append(rewards_sum)
            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")

        return reward_per_episode
