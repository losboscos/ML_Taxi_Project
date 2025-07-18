import math, random
from collections import namedtuple, deque
from itertools import count
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Replay buffer ---------------------------------------------------------
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# --- Standard DQN network (used in lecture) -------------------------------
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # outputs Q-values for all actions


# --- Hyperparameters (tuned) ---------------------------------------------
BATCH_SIZE   = 32               # smaller batches → more frequent, low-variance updates
GAMMA        = 0.985            # slightly less emphasis on very long‐term returns
EPS_START    = 1.0              # start fully exploring
EPS_END      = 0.02             # almost fully greedy at the end
NUM_EPISODES = 7_500           # more episodes so that epsilon can fully anneal
EPS_DECAY    = 12000  # decay ε over 80% of training
LR           = 3e-4             # a bit smaller for extra stability
MEM_CAP      = 50_000           # larger replay buffer for richer experience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env       = gym.make("Taxi-v3")
n_states  = env.observation_space.n
n_actions = env.action_space.n

policy_net = DQN(n_states, n_actions).to(device)
optimizer   = optim.Adam(policy_net.parameters(), lr=LR)
memory      = ReplayBuffer(MEM_CAP)

steps_done = 0
episode_durations = []
episode_returns   = []
training_errors = []

def select_action(state):
    global steps_done
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1,1)
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    S = torch.cat(batch.state)
    A = torch.cat(batch.action)
    R = torch.cat(batch.reward)
    D = torch.cat(batch.done).float()
    S2 = torch.cat(batch.next_state)

    Q = policy_net(S).gather(1, A)
    with torch.no_grad():
        Q2 = policy_net(S2).max(1)[0].unsqueeze(1)
        target = R + GAMMA * Q2 * (1 - D)

    loss = F.smooth_l1_loss(Q, target)
    training_errors.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # --- Train (no render, no interactive plots) ---------------------------
    policy_net.train()
    best_reward = float('-inf')  # inizializza best model

    for ep in range(1, NUM_EPISODES+1):
        s_int, _ = env.reset()
        state = F.one_hot(torch.tensor([s_int], device=device), n_states).float()
        total_r = 0

        for t in count():
            action = select_action(state)
            n_int, reward, term, trunc, _ = env.step(action.item())
            total_r += reward
            done = term or trunc

            next_state = F.one_hot(torch.tensor([n_int], device=device), n_states).float()
            if done:
                next_state = torch.zeros_like(next_state)

            memory.push(
                state, action, next_state,
                torch.tensor([[reward]], device=device),
                torch.tensor([[done]], device=device)
            )
            state = next_state

            optimize_model()

            if done:
                episode_durations.append(t+1)
                episode_returns.append(total_r)
                break

        # Save best model
        if ep >= 500:
            avg_ret = sum(episode_returns[-500:]) / 500
            if avg_ret > best_reward:
                best_reward = avg_ret
                torch.save(policy_net.state_dict(), "best_model.pth")
                print(f"✓ Saved new best model at episode {ep} (avg reward: {avg_ret:.2f})")

        if ep % 500 == 0:
            avg_dur = sum(episode_durations[-500:]) / 500
            print(f"Ep {ep}: avg dur={avg_dur:.1f}, avg ret={avg_ret:.1f}")

    print("Training complete\n")

    # Load best model for testing
    policy_net.load_state_dict(torch.load("best_model.pth"))

    # --- Final combined plot: durations, returns, TD error -------------------
    import os
    os.makedirs("plots", exist_ok=True)

    fig, axs = plt.subplots(ncols=3, figsize=(18, 5))

    axs[0].set_title("Episode Durations")
    axs[0].plot(episode_durations, label="Durations")
    if len(episode_durations) >= 100:
        ma_dur = [sum(episode_durations[i-99:i+1])/100 for i in range(99, len(episode_durations))]
        axs[0].plot(range(99, len(episode_durations)), ma_dur, label="100-epi MA")
    axs[0].legend()

    axs[1].set_title("Episode Returns")
    axs[1].plot(episode_returns, alpha=0.3, label="Returns")
    if len(episode_returns) >= 100:
        ma_ret = [sum(episode_returns[i-99:i+1])/100 for i in range(99, len(episode_returns))]
        axs[1].plot(range(99, len(episode_returns)), ma_ret, label="100-epi MA")
    axs[1].legend()

    axs[2].set_title("TD Error (Loss) over Time")
    axs[2].plot(training_errors, alpha=0.4, label="TD Error")
    if len(training_errors) >= 100:
        ma_td = [sum(training_errors[i - 99:i + 1]) / 100 for i in range(99, len(training_errors))]
        axs[2].plot(range(99, len(training_errors)), ma_td, label="100-step MA")
    axs[2].set_xlabel("Training Steps")
    axs[2].set_ylabel("Smooth L1 Loss")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("plots/dqn_combined_metrics.png", dpi=200, bbox_inches="tight")
    plt.show()

    # --- Quick test without rendering overhead ----------------------------
    policy_net.eval()
    test_env = gym.make("Taxi-v3", render_mode="human")
    test_returns = []
    for _ in range(5):
        MAX_TEST_STEPS = 200
        for epi in range(1, 6):
            s_int, _ = test_env.reset()
            state = F.one_hot(torch.tensor([s_int], device=device), n_states).float()
            tot_r = 0.0
            for t in range(MAX_TEST_STEPS):
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1).view(1,1)
                n_int, reward, term, trunc, _ = test_env.step(action.item())
                tot_r += reward
                done = term or trunc
                state = F.one_hot(torch.tensor([n_int], device=device), n_states).float()
                if done:
                    break
            else:
                print(f"  › Test Ep {epi}: reached {MAX_TEST_STEPS} steps without finishing.")
            test_returns.append(tot_r)
    print("Test returns:", test_returns, "avg:", sum(test_returns)/len(test_returns))
    test_env.close()
