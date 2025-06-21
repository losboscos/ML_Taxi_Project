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
from torch.optim.lr_scheduler import StepLR

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


# --- DQN network ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.shared1 = nn.Linear(n_states, 128)
        self.shared2 = nn.Linear(128, 128)
        self.value   = nn.Linear(128, 1)
        self.adv     = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        v = self.value(x)
        a = self.adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))


# --- Hyperparameters --------------------------------------------------------
BATCH_SIZE   = 64
GAMMA        = 0.995
EPS_START    = 1.0
EPS_END      = 0.05
NUM_EPISODES = 7500
EPS_DECAY    = NUM_EPISODES * 0.9
LR           = 3e-4
MEM_CAP      = 20000
TARGET_UPDATE = 1000  # non puoi, single NN -> ignoralo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env       = gym.make("Taxi-v3")
n_states  = env.observation_space.n
n_actions = env.action_space.n

policy_net = DQN(n_states, n_actions).to(device)
optimizer   = optim.Adam(policy_net.parameters(), lr=LR)
scheduler   = StepLR(optimizer, step_size=2000, gamma=0.5)
memory      = ReplayBuffer(MEM_CAP)

steps_done = 0
episode_durations = []
episode_returns   = []


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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # --- Train (no render, no interactive plots) ---------------------------
    policy_net.train()
    for ep in range(1, NUM_EPISODES+1):
        s_int, _ = env.reset()
        state = F.one_hot(torch.tensor([s_int], device=device), n_states).float()
        total_r = 0

        total_reward = 0
        for t in count():
            action = select_action(state)
            n_int, reward, term, trunc, _ = env.step(action.item())
            total_r += reward
            done = term or trunc

            next_state = F.one_hot(torch.tensor([n_int], device=device),
                                   n_states).float()
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

        scheduler.step()

        # stampa davvero ogni 500 ep
        if ep % 500 == 0:
            avg_dur = sum(episode_durations[-500:]) / 500
            avg_ret = sum(episode_returns[-500:])   / 500
            print(f"Ep {ep}: avg dur={avg_dur:.1f}, avg ret={avg_ret:.1f}")

    print("Training complete\n")

    # --- Final plots (una volta sola) --------------------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Durations")
    plt.plot(episode_durations, label="dur")
    ma = [sum(episode_durations[i-99:i+1])/100 for i in range(99, len(episode_durations))]
    plt.plot(range(99, len(episode_durations)), ma, label="100-epi MA")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Returns")
    plt.plot(episode_returns, alpha=0.3, label="return")
    ma2 = [sum(episode_returns[i-99:i+1])/100 for i in range(99, len(episode_returns))]
    plt.plot(range(99, len(episode_returns)), ma2, label="100-epi MA")
    plt.legend()

    plt.tight_layout()
    # salva i PNG
    import os
    os.makedirs("plots", exist_ok=True)
    plt.figure(1)
    plt.savefig("plots/dqn_durations.png", dpi=200, bbox_inches="tight")
    plt.figure(2)
    plt.savefig("plots/dqn_returns.png", dpi=200, bbox_inches="tight")
    plt.show()

    # --- Quick test without rendering overhead ----------------------------
    policy_net.eval()
    test_env = gym.make("Taxi-v3", render_mode="human")  # text-only
    test_returns = []
    for _ in range(5):
        s_int, _ = test_env.reset()
        state = F.one_hot(torch.tensor([s_int], device=device), n_states).float()
        tot_r = 0
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(state).argmax(dim=1).view(1,1)
            n_int, reward, term, trunc, _ = test_env.step(action.item())
            done = term or trunc
            tot_r += reward
            state = F.one_hot(torch.tensor([n_int], device=device), n_states).float()
        test_returns.append(tot_r)
    print("Test returns:", test_returns, "avg:", sum(test_returns)/len(test_returns))

    test_env.close()
