import warnings

warnings.filterwarnings("ignore")

import importlib.util

from stable_baselines3 import PPO

from myosuite.agents.utils import ActionSpaceWrapper
from myosuite.utils import gym

spec = importlib.util.find_spec("myosuite")
print(spec.origin)

# max episode steps is 300 for the table tennis environment - init.py from myosuite/envs/myo/myochallenge/__init__.py
env = gym.make(
    "myoChallengeTableTennisP1-v0", max_episode_steps=300
)  # create a training environment
env = ActionSpaceWrapper(env)

print(f"Max episode steps: {env.spec.max_episode_steps}")

obs, info = env.reset()  # reset the environment, creates a new episode

model = PPO("MlpPolicy", env, verbose=0, device="cpu")
# model = PPO.load(
#     "data/baseline_policy_torso_up_paddle_hold", env=env, device="cpu"
# )

# to train to convergence use more iterations e.g.
model.learn(total_timesteps=1e6, progress_bar=True)

# Save the agent
model.save("data/baseline_policy")
