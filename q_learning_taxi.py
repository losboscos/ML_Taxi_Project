import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3', render_mode = "human")

observation, info = env.reset()
observation = env.unwrapped.decode(state_int)


episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()