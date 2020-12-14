
import numpy as np
import time
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
import torch as th

from envs import halite_env
from envs.attraction_env import AttractionEnv
from envs.metrics_env import MetricsEnv

from models import Illogical

bot_name = 'AttractionBotOBL'
weights_name = f'sac_{bot_name}_model'

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', help='Continue')
args = parser.parse_args()

env = halite_env.Env(enemy_bot='/home/vova/Documents/multiagent_systems/L3/bots/venv3.8/bin/python /home/vova/Documents/multiagent_systems/L3/bots/Covid-chance/MyBot-v0.1.1-alpha.py')
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", bot_name=bot_name)
env = AttractionEnv(env)
env = Monitor(env)
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=[500, 500, 200])
model = SAC('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f'./runs/{time.time()}/')

# if args.load:
#     actor.load_weights(weights_name + '_final_actor.h5f')
#     critic.load_weights(weights_name + '_final_critic.h5f')


nb_steps = 60_000
model.learn(total_timesteps=nb_steps, log_interval=1)
model.save(weights_name)

# agent.test(env, nb_episodes=1, visualize=False)
