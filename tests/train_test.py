from src.Train import Train
from matplotlib import pyplot as plt

trainer = Train("Sarsa", "MountainCar-v0", rbf_order=6)

reward_per_episode, steps_per_episode = trainer.train(episodes=1000, alpha=25e-3, gamma=1, epsilon=0.1, max_steps=350, render=True)


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
