
import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# sys.stderr = open('errors.log', 'w')

import time
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import hlt
from envs import halite_env
from envs.attraction_env import AttractionEnv
from models import Illogical

parser = argparse.ArgumentParser()
parser.add_argument('--use-stdio', action='store_true', help='Use stdio for game')
args = parser.parse_args()

def ifprint(*args_, **kwargs):
    if not args.use_stdio:
        print(*args_, **kwargs)

bot_name = 'AttractionBot'

env = halite_env.Env(stdio=args.use_stdio)
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", bot_name=bot_name)
env = AttractionEnv(env)

actor = Illogical.make_model_actor(env)
actor.summary(print_fn=ifprint)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_location = os.path.join(current_directory, f'ddpg_{bot_name}_weights_final_actor.h5f')
actor.load_weights(model_location)

total_reward = 0
reward = 0
observations = env.reset()
while True:
    actions = actor.predict(observations.reshape((1,) + env.observation_space.shape))[0]

    ifprint(f'{actions=}')

    observations, reward, done, _ = env.step(actions)

    total_reward += reward
    ifprint(f'{reward=}')

    if done:
        break

env.close()
ifprint(f'done, {total_reward=}')
