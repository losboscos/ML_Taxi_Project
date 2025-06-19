import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("Taxi-v3")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# env.reset(seed=seed)
# env.action_space.seed(seed)
# env.observation_space.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = env.observation_space.n

net = DQN(n_observations, n_actions).to(device)

optimizer = optim.AdamW(net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(durations, dtype=torch.float)
    plt.clf()
    plt.title('Result' if show_result else 'Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepara tensori batched
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    # Q(s,a) corrente
    state_action_values = net(state_batch).gather(1, action_batch)

    # Calcolo del target: r + γ max_a' Q(s', a') usando LA STESSA policy_net
    with torch.no_grad():
        next_state_values = net(next_state_batch).max(1)[0].unsqueeze(1)
        # se done, valore futuro = 0
        next_state_values = next_state_values * (1 - done_batch)

        expected_state_action_values = reward_batch + GAMMA * next_state_values

    # Loss e backprop
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # numero di episodi più lungo se ho GPU/MPS, altrimenti breve per debug
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 600

    for i_episode in range(num_episodes):
        # 1) reset dell'ambiente e preparazione del tensore stato
        state_int, _ = env.reset()
        state = torch.nn.functional.one_hot(
            torch.tensor([state_int], device=device), num_classes=env.observation_space.n
        ).float()  # → shape [1, n_states]

        for t in count():
            # 2) selezione azione ε-greedy
            action = select_action(state)

            # 3) interazione con l'ambiente
            next_state_int, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # 4) prepara il next_state (o None se done)
            if done:
                next_state = torch.zeros_like(state)
            else:
                next_state = torch.nn.functional.one_hot(
                    torch.tensor([next_state_int], device=device),
                    num_classes=env.observation_space.n
                ).float()

            # 5) memorizza la transizione completa (state, action, reward, next_state, done)
            memory.push(state, action,
            next_state,                                  # 3) next_state
            torch.tensor([[reward]], device=device, dtype=torch.float32),  # 4) reward
            torch.tensor([[done]],   device=device, dtype=torch.float32))  # 5) done

            # 6) avanza lo stato
            state = next_state

            # 7) ottimizzazione di un passo sulla stessa rete `net`
            optimize_model()

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    print('Complete')
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()
    plt.ioff()
    plt.show()

