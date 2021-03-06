import random

import numpy as np
from src.Agents.SARSA.Sarsa import Sarsa
from src.Agents.SARSA.DeepSARSA import DeepSARSA
from src.Agents.DQN.DQN import DQN

import pickle


class Train:

    def __init__(self, env, model_name, model_config):
        """
        Initializes the model with given configuration to be trained on the given environment
        :param env: Training environment
        :param model_name: Model to be trained
        :param model_config: Configuration for the model
        """

        models = {"Sarsa": Sarsa, "DeepSARSA": DeepSARSA, "DQN": DQN}

        self.env = env

        if model_name is "Sarsa"or "DeepSARSA":
            self.model_type = "SARSA"
        elif model_name is "DQN":
            self.model_type = "DQN"

        print(f"State Space: {self.env.observation_space} \nAction Space: {self.env.action_space}")


        # todo: do some box/discrete checking here later

        self.model = models[model_name](**model_config)

    def play_sarsa(self, episodes, max_steps=None, render=False):

        for ep in range(episodes):

            # get initial state and action (via policy)
            state = self.env.reset()
            i = 1
            rewards_sum = 0

            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()


                action = self.greedy_policy(state, 0)
                next_state, reward, terminate, info = self.env.step(action)  # commit to action
                state = next_state

                # to find average reward for the episode, hold a sum of all rewards and the steps taken
                i += 1
                rewards_sum += reward

                if terminate is True or i > max_steps:  # in termination, update using only current state-action, and break out of this episode
                    break

                # update current state-action



            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")


    def train(self, episodes, learning_rate, discount, epsilon, max_steps=None, decay=False, render=False):
        """
        Training function, loops for episodes, implemented using Sutton's RL book
        :param episodes: trial amount
        :param learning_rate: learning rate
        :param discount: discount, generally 1
        :param epsilon: the exploration rate, small or decaying
        :param max_steps: max steps per episode
        :param decay: condition on whether to decay exploration and learning rate or not
        :param render: condition which determines whether the game gets rendered or not
        :return: steps/reward per episode
        """

        reward_per_episode = []
        steps_per_episode = []

        for ep in range(episodes):

            # get initial state and action (via policy)
            state = self.env.reset()
            action = self.greedy_policy(state, epsilon)

            i = 1
            rewards_sum = 0

            if decay is True and epsilon > 0.1:  # decay
                epsilon *= 0.995
                # learning_rate *= 0.99995


            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()

                next_state, reward, terminate, info = self.env.step(action)  # commit to action

                # to find average reward for the episode, hold a sum of all rewards and the steps taken
                i += 1
                rewards_sum += reward

                if terminate is True or (max_steps is not None and i > max_steps):  # in termination, update using only current state-action, and break out of this episode
                    self.model.update(state, action, reward, learning_rate=learning_rate, discount=discount, terminate=True)
                    break

                if self.model_type is "SARSA":
                    # else get next action using the next state, update following the SARSA rule
                    next_action = self.greedy_policy(next_state, epsilon)
                self.model.update(state, action, reward, next_state=next_state, next_action=next_action,
                                  learning_rate=learning_rate, discount=discount)

                # update current state-action

                state = next_state
                action = next_action

            reward_per_episode.append(rewards_sum)
            steps_per_episode.append(i)

            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")



        return reward_per_episode, steps_per_episode


    def train_dqn(self, episodes, time_steps, C, min_replay_count, batch_size, learning_rate, discount, epsilon, epsilon_decay=0.95, lr_decay = 0.99, duration=30, lr_low=0, max_steps=None, decay=False, render=False):
        """
        Training function, loops for episodes, implemented using Sutton's RL book
        :param episodes: trial amount
        :param time_steps: number of time steps
        :param C: target network parameters are updated every C steps
        :param min_replay_count: Experience replay updates happen after this count
        :param batch_size: Size of the mini batch for experience replay update
        :param learning_rate: learning rate
        :param discount: discount, generally 1
        :param epsilon: the exploration rate, small or decaying
        :param max_steps: max steps per episode
        :param decay: condition on whether to decay exploration and learning rate or not
        :param render: condition which determines whether the game gets rendered or not
        :return: steps/reward per episode
        """

        reward_per_episode = []
        steps_per_episode = []
        loss_per_episode = []

        lrinit = learning_rate


        target_update_count = 0
        for ep in range(episodes):

            # get initial state and action (via policy)
            state = self.env.reset()


            i = 1
            rewards_sum = 0

            if decay is True and epsilon > 0.01 and learning_rate > lr_low:  # decay
                #epsilon *= epsilon_decay
                # print("epsilon", epsilon)
                #learning_rate *= lr_decay # landerda yok

                epsilon -= 0.99 / 300
                learning_rate -= 0.00015 / 300

            loss = 0
            #print(ep, "ep")


            for t in range(time_steps):
                #print(t)
                target_update_count += 1

                if render is True and ep > 1450:  # for visualization
                    self.env.render()

                # With probability epsilon select a random action a_t, otherwise select at = max_a Q(s_t, a; ??)
                action = self.greedy_policy(state, epsilon)
                # print("action")
                # print(action)


                # Execute action at in emulator and observe reward r_t and s_t
                next_state, reward, terminate, info = self.env.step(action)  # commit to action


                # Store transition in replay memory
                self.model.replay_memory.append([state, action, reward, next_state, terminate])  # s_t, a_t, r_t, s_(t+1)'

                state = next_state

                # to find average reward for the episode, hold a sum of all rewards and the steps taken
                i += 1
                rewards_sum += reward

                if terminate:
                    break

                # Sample random mini batch from replay memory
                if len(self.model.replay_memory) < min_replay_count:
                    continue



                #print("updating at", ep)
                mini_batch = random.sample(self.model.replay_memory, batch_size)

                loss += self.model.update(mini_batch, learning_rate, discount)


                # Update target network weights
                if target_update_count % C == 0:
                    self.model.update_target()



            loss_per_episode.append(loss/t)

            reward_per_episode.append(rewards_sum)
            steps_per_episode.append(i)

            if (ep % 1 == 0):
                print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")

            if (ep % 10 == 0 and ep < 300):
                print("epsilon", epsilon)



        # Save model



        return reward_per_episode, steps_per_episode, loss_per_episode



    def greedy_policy(self, state, epsilon):
        """
        Greedy policy, more explanation on the report
        :param state: state vector
        :param epsilon: exploration rate
        :return: next action
        """

        if np.random.rand() > epsilon:
            # print("state", state)
            # print("self.model.q_approx(state)")
            # print(self.model.q_approx(state))
            approx = self.model.q_approx(state)
            index = np.argmax(approx)
            return index
        else:
            return self.env.action_space.sample()


    def max_action(self, state):
        """
        Returns action that maximizes q function
        :param state: state vector
        :return: next action
        """
        return np.argmax(self.model.q_approx(state))

