import gym
import numpy as np
from src.Agents.Sarsa import Sarsa
from src.FeatureConstructors.RadialBasis import RBF





class Train:

    def __init__(self, model_name, env_name, rbf_order):

        models = {"Sarsa": Sarsa}

        self.env = gym.make(env_name).env

        print(f"State Space: {self.env.observation_space} \nAction Space: {self.env.action_space}")


        # todo: do some box/discreet checking here later

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        state_low = self.env.observation_space.low
        state_high= self.env.observation_space.high

        basis_function = RBF(rbf_order, state_dim, state_low, state_high)


        self.model = models[model_name](state_dim, action_dim, basis_function)



    def train(self, episodes, alpha, gamma, epsilon, max_steps=None, render=False):

        reward_per_episode = []
        steps_per_episode = []

        for ep in range(episodes):

            # get initial state and action (via policy)
            state = self.env.reset()
            action = self.greedy_policy(state, epsilon)

            i = 1
            rewards_sum = 0

            while True:  # loop controlled with termination of state, run until done

                if render is True:  # for visualization
                    self.env.render()

                next_state, reward, terminate, info = self.env.step(action)  # commit to action

                # to find average reward for the episode, hold a sum of all rewards and the steps taken
                i += 1
                rewards_sum += reward

                if terminate is True or (max_steps is not None and i > max_steps):  # in termination, update using only current state-action, and break out of this episode
                    self.model.update(state, action, reward, alpha=alpha, gamma=gamma, terminate=True)
                    break

                # else get next action using the next state, update following the SARSA rule
                next_action = self.greedy_policy(next_state, epsilon)
                self.model.update(state, action, reward, next_state=next_state, next_action=next_action, alpha=alpha, gamma=gamma)

                # update current state-action

                state = next_state
                action = next_action

            reward_per_episode.append(rewards_sum)
            steps_per_episode.append(i)

            print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")

        return reward_per_episode, steps_per_episode



    def greedy_policy(self, state, epsilon):

        if np.random.rand() > epsilon:
            return np.argmax(self.model.q_approx(state))
        else:
            return self.env.action_space.sample()













