
class RandomAgent(object):

    def __init__(self, env):

        self.env = env


    def play(self, episodes, max_steps, reward_per_episode, render=False):

            for ep in range(episodes):
                # get initial state and action (via policy)
                state = self.env.reset()
                i = 1
                rewards_sum = 0

                while True:  # loop controlled with termination of state, run until done

                    if render is True:  # for visualization
                        self.env.render()

                    action = self.env.action_space.sample()
                    next_state, reward, terminate, info = self.env.step(action)  # commit to action

                    i += 1
                    rewards_sum += reward

                    if terminate or i > max_steps:
                        break

                reward_per_episode.append(rewards_sum)
                if ep % 100 == 0:
                    print(f"Episode ended: {ep}\nReward total: {rewards_sum}\nSteps: {i}\n")

            return reward_per_episode
