import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from Agent.q_learning_taxi_agent import TaxiAgent
from tqdm import tqdm

def run(learning_rate, n_episodes, start_epsilon, epsilon_decay, final_epsilon):
    env = gym.make('Taxi-v3', render_mode = None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = TaxiAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    #rewards_per_episode = np.zeros(n_episodes) #aggiunta questa cosa, inizializzo tutti a zero i rewards degli espisodi
    for episode in tqdm(range(n_episodes)):

        state_int, info = env.reset()

        rewards = 0
        episode_over = False
        while not episode_over:
            action = agent.get_action(env, state_int)  # agent policy that uses the observation and info
            next_state_int, reward, terminated, truncated, info = env.step(action)
            #print(f" reward from this single env.step: {reward}")

            rewards += reward

            # update the agent
            agent.update(state_int, action, reward, terminated, next_state_int)

            episode_over = terminated or truncated 
            state_int = next_state_int
        #print(rewards)
        agent.decay_epsilon()
        #rewards_per_episode[episode] = rewards # aggiunta anche qui
    return env, agent

   

def visualize_training(env, agent):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Reward rolling average
    axs[0].set_title("Episode rewards")
    rewards = np.array(env.return_queue)
    reward_ma = np.convolve(rewards, np.ones(rolling_length), mode="valid") / rolling_length
    axs[0].plot(reward_ma)

    # Length rolling average
    axs[1].set_title("Episode lengths")
    lengths = np.array(env.length_queue)
    length_ma = np.convolve(lengths, np.ones(rolling_length), mode="same") / rolling_length
    axs[1].plot(length_ma)

    # Training error rolling average
    axs[2].set_title("Training Error")
    errors = np.array(agent.training_error)
    error_ma = np.convolve(errors, np.ones(rolling_length), mode="same") / rolling_length
    axs[2].plot(error_ma)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # qui definisci i valori dei tuoi hyperparametri
    learning_rate = 0.1
    n_episodes = 20_000
    start_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_decay = (start_epsilon - final_epsilon) / (n_episodes * 0.8)
    
    env, agent = run(
        learning_rate=learning_rate,
        n_episodes=n_episodes,
        start_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    visualize_training(env, agent)
    env.close()
