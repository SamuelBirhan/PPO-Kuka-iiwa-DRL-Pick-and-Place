import time  # Add this import at the top
from tensorboardX import SummaryWriter
import timeit
from datetime import timedelta
import torch
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import utils
from model import ActorCritic

from utils import calc_returns
from custom_env import env, device
import torch.optim as optim

from train_kuka_gpu import collect_trajectories
from model import ActorCritic

# Initialize values
init_screen = utils.get_screen()
_, _, screen_height, screen_width = init_screen.shape
action_size = env.action_space.shape[0]

writer = SummaryWriter()
i_episode = 0
ten_rewards = 0

# ActorCritic model and optimizer
policy = ActorCritic(state_size=(screen_height, screen_width),
                     action_size=action_size,
                     shared_layers=[128, 64],
                     critic_hidden_layers=[64],
                     actor_hidden_layers=[64],
                     init_type='xavier-uniform',
                     seed=0).to(device)
optimizer = optim.Adam(policy.parameters(), lr=2e-4)

# Paths
path = 'all_policy_ppo_neww.pt'
PATH = 'policy_ppo_neww.pt'

# Training parameters
scores_window = deque(maxlen=100)
opt_epoch = 5
season = 1000000
batch_size = 40
tmax = 400

epsilons, betas = [], []  # Track epsilon and beta values
entropy_values, value_losses = [], []  # Track entropy and value losses
save_scores, mean_rewards, seasons = [], [], []
best_mean_reward = -float('inf')

# Load checkpoint
checkpoint = torch.load(path)
policy.load_state_dict(checkpoint['policy_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
seasons = checkpoint['seasons']
mean_rewards = checkpoint['mean_rewards']
save_scores = checkpoint['save_scores']
betas = checkpoint['betas']
epsilons = checkpoint['epsilons']
best_mean_reward = np.max(mean_rewards)

# Best season reward tracking
best_season_reward = -float('inf')

# Plot Beta Decay and Epsilon Decay
plt.figure(figsize=(10, 6))

# Beta Decay
plt.plot(seasons, betas, label='Beta Decay', linestyle='-', linewidth=2)

# Epsilon Decay
plt.plot(seasons, epsilons, label='Epsilon Decay', linestyle='--', linewidth=2)

plt.title("Beta and Epsilon Decay Over Seasons")
plt.xlabel("Seasons")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()


for s in range(seasons[-1] + 1, season + 1):
    season_start_time = time.time()
    policy.eval()
    old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = collect_trajectories(env, policy, tmax)

    # Calculate reward and append scores
    season_score = rewards_lst.sum(dim=0).item()
    scores_window.append(season_score)
    save_scores.append(season_score)
    seasons.append(s)

    if season_score > best_season_reward:
        best_season_reward = season_score
        torch.save({'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mean_rewards': mean_rewards,
                    'save_scores': save_scores,
                    'seasons': seasons}, "Best_season_model.pt")
        print(f"New best season reward {season_score:.3f} achieved! Model saved as 'Best_season_model.pt'.")

    # Compute GAE and target returns
    gea, target_value = calc_returns(rewards=rewards_lst, values=values_lst, dones=dones_list)
    gea = (gea - gea.mean()) / (gea.std() + 1e-8)
    policy.train()

    # Concatenate data
    old_probs_lst = old_probs_lst.reshape(-1)
    states_lst = states_lst.reshape(-1, *states_lst.shape[2:])
    actions_lst = actions_lst.reshape(-1, action_size)
    gea = gea.reshape(-1)
    target_value = target_value.reshape(-1)

    # Optimize policy
    n_sample = len(old_probs_lst) // batch_size
    idx = np.arange(len(old_probs_lst))
    np.random.shuffle(idx)

    for epoch in range(opt_epoch):
        for b in range(n_sample):
            ind = idx[b * batch_size:(b + 1) * batch_size]
            g, tv = gea[ind], target_value[ind]
            actions, old_probs = actions_lst[ind], old_probs_lst[ind]
            action_est, values = policy(states_lst[ind])

            # Calculate entropy, log_probs, and loss components
            sigma = nn.Parameter(torch.zeros(action_size))
            dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
            ratio = torch.exp(log_probs - old_probs)

            L_CLIP = torch.mean(torch.min(ratio * g, torch.clamp(ratio, 1 - epsilons[-1], 1 + epsilons[-1]) * g))
            L_VF = 0.5 * (tv - values).pow(2).mean()
            loss = -(L_CLIP - 0.5 * L_VF + betas[-1] * entropy.mean())

            # Append entropy and value loss for monitoring
            entropy_values.append(entropy.mean().item())
            value_losses.append(L_VF.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()

    # Decay epsilon and beta
    epsilons.append(epsilons[-1] * 0.999 if epsilons else 0.2)
    betas.append(betas[-1] * 0.998 if betas else 0.01)

    # Save progress
    mean_rewards.append(np.mean(scores_window))
    torch.save({'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_rewards': mean_rewards,
                'save_scores': save_scores,
                'seasons': seasons,
                'betas': betas,
                'epsilons': epsilons}, path)
    print(f"Season {s}, Score: {season_score}, Beta: {betas[-1]:.5f}, Epsilon: {epsilons[-1]:.5f}")

# PLOT RESULTS
plt.figure()
plt.plot(seasons, betas, label='Beta')
plt.plot(seasons, epsilons, label='Epsilon')
plt.title('Beta and Epsilon Decay')
plt.xlabel('Season')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(seasons, save_scores, alpha=0.5, label='Reward per Season')
plt.plot(seasons, mean_rewards, label='Mean Reward')
plt.title('Rewards Over Seasons')
plt.xlabel('Season')
plt.ylabel('Reward')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(range(len(entropy_values)), entropy_values, label='Policy Entropy')
plt.title('Policy Entropy Over Time')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(range(len(value_losses)), value_losses, label='Value Function Loss')
plt.title('Value Function Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Action Distribution
actions = actions_lst.cpu().numpy()
plt.figure()
plt.hist(actions[:, 0], bins=20, alpha=0.5, label='Action 1')
plt.hist(actions[:, 1], bins=20, alpha=0.5, label='Action 2')
plt.hist(actions[:, 2], bins=20, alpha=0.5, label='Action 3')
plt.title('Action Distribution')
plt.xlabel('Action Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()

writer.close()
print("Training complete and plots generated.")
