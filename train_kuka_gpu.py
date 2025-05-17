import torch
import torch.nn as nn
import numpy as np
from model import ActorCritic
from utils import get_screen, calc_returns
from custom_env import env, device
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# Set device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

env.reset()

# Number of agents
# num_agents = 1
# print('Number of agents:', num_agents)
#
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
#
# # Size of each action
action_size = env.action_space.shape[0]
# print('Size of each action:', action_size)
#
# plt.figure()
# plt.imshow(init_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

writer = SummaryWriter()
i_episode = 0
ten_rewards = 0


def collect_trajectories(envs, policy, tmax, nrand=5):
    global i_episode
    global ten_rewards
    global writer

    # Initialize returning lists and start the game
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []
    value_list = []
    done_list = []

    state = envs.reset()

    for t in range(tmax):
        states = get_screen().to(device)
        action_est, values = policy(states)
        sigma = nn.Parameter(torch.zeros(action_size, device=device))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()

        env_actions = actions.cpu().numpy()
        _, reward, done, _ = envs.step(env_actions[0])
        #print(f"Reward in each step: {reward}")
        rewards = torch.tensor([reward], device=device)

        dones = torch.tensor([done], device=device)

        state_list.append(states.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()):
            ten_rewards += reward
            i_episode += 1
            state = envs.reset()
            if i_episode % 10 == 0:
                writer.add_scalar('ten episodes average rewards', ten_rewards / 10.0, i_episode)
                ten_rewards = 0

    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list


def calc_returns(rewards, values, dones):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    # Create empty buffer
    GAE = torch.zeros(n_step, n_agent).float().to(device)
    returns = torch.zeros(n_step, n_agent).float().to(device)

    # Set start values
    GAE_current = torch.zeros(n_agent).float().to(device)

    TAU = 0.95
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1. - dones[irow].float())

        # Calculate TD Error
        td_error = rewards_current + gamma * values_next - values_current
        # Update GAE, returns
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current + gamma * returns_current
        # Set GAE, returns to buffer
        GAE[irow] = GAE_current
        returns[irow] = returns_current

        values_next = values_current

    return GAE, returns


def eval_policy(env, policy, tmax):
    """Evaluates the policy by running it in the environment for a set number of timesteps."""
    rewards_list = []
    state = env.reset()
    for t in range(tmax):
        states = get_screen().to(device)
        action_est, _ = policy(states)

        actions = torch.clamp(action_est, -1.0, 1.0)
        _, reward, done, _ = env.step(actions[0].cpu().numpy())
        print(f"Reward Evaluation: {reward}")

        rewards_list.append(reward)
        if done:
            break
    return rewards_list
