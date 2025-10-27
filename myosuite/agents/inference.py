import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

import numpy as np
import skvideo.io
from stable_baselines3 import PPO
from tqdm import tqdm

import myosuite
from myosuite.agents.utils import custom_action
from myosuite.utils import gym

env = gym.make('myoChallengeTableTennisP1-v0', max_episode_steps=300) # create a training environment 
obs, info = env.reset() # reset the environment, creates a new episode


model = PPO.load("data/baseline_policy", device='cpu')

front_frames = [] # placeholder
side_frames = []
all_rewards = [] # placeholder for all rewards
data_store = [] # store the data

num_episodes = 1

# calculate the reward from 10 epsiode
hit_debug = []
all_paths = []
for n_episode in tqdm(range(num_episodes)):
  print(f"Episode: {n_episode}")
  env.reset()
  ep_rewards = [] # placeholder for episodic rewards

  terminated = False
  truncated = False
  step = 0
  
  paths = []
  while not (terminated or truncated):
    step += 1
    o = env.get_obs() # get observation from environment
    a = model.predict(o)[0] # predict the action based on the observation
    a = custom_action(env, a)
    
    next_o, r, terminated, truncated, info = env.step(a)  # take an action based on the current observation
    data_store.append({"action": a.copy(),
              "act":env.unwrapped.sim.data.act.copy(),
              "reward":r})
    ep_rewards.append(r)
    hit_debug.append(info['rwd_dict'])
    if n_episode == num_episodes-1: # only save frames for the last episode
        front_frames.append(env.sim.renderer.render_offscreen(width=640, height=480, camera_id=1))
        side_frames.append(env.sim.renderer.render_offscreen(width=640, height=480, camera_id=2))

    paths.append({"env_infos": info})
    all_paths.append({"env_infos": info})
  all_rewards.append(np.sum(ep_rewards))
  print("Episode Reward:", all_rewards[-1])
  
  # Get score per episode
  metrics = env.get_metrics(paths)
  print(f"Episode {n_episode} Score: {metrics['score']}, Effort: {metrics['effort']}")
  
  rwd_history = env.get_rwd_history().copy()
  output_rwd_history = {}
  for key in rwd_history.keys():
    if key in ['reach_dist', 'palm_dist', 'paddle_quat', 'act_reg']:
      output_rwd_history[key] = sum([val[0][0] for val in rwd_history[key]])
    else:
      output_rwd_history[key] = sum(rwd_history[key])
  print(output_rwd_history)

# Get score for all episodes
print("All Episodes Score:")
metrics = env.get_metrics(all_paths)
print(f"Score: {metrics['score']}, Effort: {metrics['effort']}")

env.close()
# # make a local copy
skvideo.io.vwrite(f'myoChallengePingPong_front_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4', np.asarray(front_frames),inputdict = {'-r': '100'}, outputdict={"-pix_fmt": "yuv420p"})
skvideo.io.vwrite(f'myoChallengePingPong_side_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4', np.asarray(side_frames),inputdict = {'-r': '100'}, outputdict={"-pix_fmt": "yuv420p"})