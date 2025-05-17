import sys
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.optim as optim

# Import and patch gym for compatibility
import gym
from collections import UserDict

registry = UserDict(gym.envs.registration.registry)
if not hasattr(registry, "env_specs"):  # Compatibility fix for newer Gym versions
    registry.env_specs = registry
gym.envs.registration.registry = registry

# Import PyBullet
import pybullet_envs
import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

# Initialize the Kuka Diverse Object Environment
class CustomKukaEnv(KukaDiverseObjectEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graspSuccess = 0
        self._attempted_grasp = False  # Track whether a grasp has been attempted.

    def step(self, action):
        """Override step to ensure reward precision."""
        obs, reward, done, info = super().step(action)
        reward = round(float(reward), 3)  # Explicitly round to 3 decimals
        return obs, reward, done, info

    def _reward(self):
        """Calculates the reward for the episode."""
        reward = 0
        self._graspSuccess = 0

        # Check if any object is above height 0.2
        for uid in self._objectUids:
            pos, _ = p.getBasePositionAndOrientation(uid)
            if pos[2] > 0.2:
                self._graspSuccess += 1
                reward = 1  # Success reward
                break

        # Additional components
        contact_points = p.getContactPoints(self._kuka.kukaUid)
        contact_reward = 0.5 if len(contact_points) > 0 else 0
        step_penalty = -0.01
        success_reward = 1 if self._attempted_grasp and self._gripper_holding_object() else 0

        reward = 0.3 * contact_reward + success_reward + step_penalty

        return round(reward, 3)  # Round the final reward

    def _gripper_holding_object(self):
        """Determines if the gripper is holding an object."""
        contact_points = p.getContactPoints(self._kuka.kukaUid)
        return len(contact_points) > 0


# Initialize environment with specific settings
env = CustomKukaEnv(renders=True, cameraRandom=0.75, isDiscrete=False, removeHeightHack=False, maxSteps=10,
                    numObjects=5)
env.cid = p.connect(p.DIRECT)
action_space = spaces.Box(low=-1, high=1, shape=(5,))

# Device configuration (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")