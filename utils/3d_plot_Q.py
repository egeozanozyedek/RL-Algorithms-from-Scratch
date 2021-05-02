import numpy as  np
from matplotlib import pyplot as plt, cm
from src.Agents.Sarsa import Sarsa
import gym

from src.FeatureConstructors.RadialBasis import RBF


def threeD_plot_q(q_function, state_low, state_high):
    div = 100
    pos = np.linspace(state_low[0], state_high[0], div)
    vel = np.linspace(state_low[1], state_high[1], div)
    grid = np.meshgrid(pos, vel)
    coordinates = np.vstack(map(np.ravel, grid)).T
    Q = -np.asarray([q_function(s) for s in coordinates])
    print(Q.shape)


    fig = plt.figure(figsize=(12, 8), dpi=160)
    ax = [fig.add_subplot(1, 1, i+1, projection='3d') for i in range(1)]

    ax[0].plot_surface(*grid,  Q.max(axis=1).reshape(div, div), cmap=cm.coolwarm)
    ax[0].set_xlabel('Position')
    ax[0].set_ylabel('Velocity')
    ax[0].set_title("Cost-to-Go Function ($-max_a q(s, a, w)$)")
    #
    # ax[1].plot_surface(*grid, -Q[:, 0].reshape(div, div), cmap=cm.coolwarm)
    # ax[1].set_xlabel('Position')
    # ax[1].set_ylabel('Velocity')
    # ax[1].set_title("Q values for accelerate left (a=0) ($q(s, a=0, w)$)")
    #
    # ax[2].plot_surface(*grid, -Q[:, 1].reshape(div, div), cmap=cm.coolwarm)
    # ax[2].set_xlabel('Position')
    # ax[2].set_ylabel('Velocity')
    # ax[2].set_title("Q values for stay still (a=1) ($q(s, a=1, w)$)")
    #
    # ax[3].plot_surface(*grid, -Q[:, 2].reshape(div, div), cmap=cm.coolwarm)
    # ax[3].set_xlabel('Position')
    # ax[3].set_ylabel('Velocity')
    # ax[3].set_title("Q values for accelerate right (a=2) ($q(s, a=2, w)$)")
    #
    plt.savefig("../figures/report1/3dplotSARSA.png",  bbox_inches = 'tight')



weights = np.load(f"/Users/egeozanozyedek/Documents/PyCharmProjects/eee485_term_project/weights/Sarsa/W.npy")

env = gym.make("MountainCar-v0").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
state_low = env.observation_space.low
state_high = env.observation_space.high



rbf_order = 6
basis_function = RBF(rbf_order, state_dim, state_low, state_high)
sarsa_config = {"state_dim": state_dim, "action_dim": action_dim, "basis_function": basis_function}
model = Sarsa(**sarsa_config)

model.W = weights

threeD_plot_q(model.q_approx, state_low, state_high)