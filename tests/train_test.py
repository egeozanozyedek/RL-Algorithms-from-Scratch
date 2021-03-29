from src.Sarsa import Sarsa
from src.Train import Train
import gym
from matplotlib import pyplot as plt

trainer = Train("Sarsa", "MountainCar-v0", rbf_order=10)

reward_per_episode, steps_per_episode = trainer.train(episodes=1000, alpha=0.025, gamma=0.995, epsilon=0.1, max_steps=None, render=False)


plt.figure(figsize=(24, 8), dpi=160)
plt.subplot(1, 2, 1)
plt.xlabel("Episode")
plt.ylabel("Reward of Episode")
plt.plot(reward_per_episode, "C2")
plt.subplot(1, 2, 2)
plt.xlabel("Episode")
plt.ylabel("Time Steps Taken")
plt.plot(steps_per_episode, "C3")
plt.show()
