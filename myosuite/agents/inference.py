import warnings
warnings.filterwarnings('ignore')

import myosuite
from myosuite.utils import gym
import skvideo.io
from tqdm import tqdm
import numpy as np
from stable_baselines3 import PPO


env = gym.make('myoChallengeTableTennisP1-v0', max_episode_steps=10000) # create a training environment 

obs, info = env.reset() # reset the environment, creates a new episode


model = PPO.load("data/baseline_policy", device='cpu')

frames = [] # placeholder
all_rewards = [] # placeholder for all rewards
data_store = [] # store the data

num_episodes = 10

# calculate the reward from 10 epsiode
for n_episode in tqdm(range(num_episodes)):
  print(f"Episode: {n_episode}")
  env.reset()
  ep_rewards = [] # placeholder for episodic rewards

  terminated = False
  truncated = False
  step = 0
  while not (terminated or truncated):
    step += 1
    o = env.get_obs() # get observation from environment
    a = model.predict(o)[0] # predict the action based on the observation
    next_o, r, terminated, truncated, info = env.step(a)  # take an action based on the current observation
    data_store.append({"action": a.copy(),
              "act":env.unwrapped.sim.data.act.copy(),
              "reward":r})
    if n_episode == num_episodes-1: # only save frames for the last episode
        frames.append(env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1))
        ep_rewards.append(r)

  all_rewards.append(np.sum(ep_rewards))
  print("Episode Reward:", all_rewards[-1])
env.close()

print(f"Average reward: {np.mean(all_rewards)} over {num_episodes} episodes")

# # make a local copy
skvideo.io.vwrite('myoChallengeSoccer_baseline.mp4', np.asarray(frames),inputdict = {'-r': '100'}, outputdict={"-pix_fmt": "yuv420p"})

# save video in the notebook
# skvideo.io.vwrite('data/myoChallengeSoccer_baseline.mp4', np.asarray(frames),inputdict = {'-r': '100'}, outputdict={"-pix_fmt": "yuv420p"})