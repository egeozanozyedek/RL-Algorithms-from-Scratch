import pickle
import numpy as np
import gym
#
from src.Agents.RandomAgent import RandomAgent
#
#
#
with open(f"../src/Agents/DDPG/saved_agents/agent0.8811363749303533", 'rb') as f:
    _, reward, _, _ = pickle.load(f)

print(np.asarray(reward)[-100:].mean())

# with open(f"dqn_breakout_loss.npy", 'rb') as f:
#     loss_ddpg = np.load(f)

print(reward_ddpg.shape)
from matplotlib import pyplot as plt

from src.Agents.RandomAgent import RandomAgent
env = gym.make("Pendulum-v0").env
ra = RandomAgent(env)
aa = []
ra.play(len(reward_per_episode), 1000, aa, False)


plt.figure(figsize=(12, 8), dpi=160)
plt.ylabel("Reward per Episode")
plt.xlabel("Episodes Taken")
plt.plot(aa, "C3", label="Random Agent")
plt.plot(reward_per_episode, "C2", label="DDPG")
plt.legend()
# plt.show()
plt.savefig("../figures/finalreport/ddpg_pendulum_reward.png",  bbox_inches = 'tight')


# plt.figure(figsize=(12, 8), dpi=160)
# plt.ylabel("Loss per Episode")
# plt.xlabel("Episodes Taken")
# plt.plot(loss_ddpg.T, "C2", label="DQN")
# plt.legend()
# plt.savefig("../figures/finalreport/dqn_breakout_loss.png",  bbox_inches = 'tight')
# from src.Agents.RandomAgent import RandomAgent
#
# env = gym.make("Breakout-v0").env
# ra = RandomAgent(env)
# aa = []
# ra.play(1500, 10000, aa, False)
# print("Breakout:", np.asarray(aa).mean())
#
#
# env = gym.make("CartPole-v0").env
# ra = RandomAgent(env)
# aa = []
# ra.play(400, 200, aa, False)
# print("CartPole:", np.asarray(aa).mean())
#
#
#
# env = gym.make("LunarLander-v2").env
# ra = RandomAgent(env)
# aa = []
# ra.play(500, 1000, aa, False)
# print("LunarLander:", np.asarray(aa).mean())
#
#
# env = gym.make("BipedalWalker-v3").env
# ra = RandomAgent(env)
# aa = []
# ra.play(2000, 700, aa, False)
# print("Bipedal:", np.asarray(aa).mean())
#
#
# env = gym.make("Pendulum-v0").env
# ra = RandomAgent(env)
# aa = []
# ra.play(100, 1000, aa, False)
# print("Pendulum:", np.asarray(aa).mean())