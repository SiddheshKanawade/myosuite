import warnings
warnings.filterwarnings('ignore')

import myosuite
from myosuite.utils import gym
import skvideo.io
from tqdm import tqdm
import numpy as np
from stable_baselines3 import PPO

import importlib.util
spec = importlib.util.find_spec("myosuite")
print(spec.origin)


env = gym.make('myoChallengeTableTennisP1-v0', max_episode_steps=10000) # create a training environment 

obs, info = env.reset() # reset the environment, creates a new episode

model = PPO("MlpPolicy", env, verbose=0, device='cpu')

# to train to convergence use more iterations e.g.
model.learn(total_timesteps=1e5, progress_bar=True)

# Save the agent
model.save("data/baseline_policy")