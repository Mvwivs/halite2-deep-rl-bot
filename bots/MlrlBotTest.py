
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.stderr = open('/dev/null', 'w')

import time
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import hlt
import halite_env
from envs.ml_rl import MlrlEnv
from models import Illogical

parser = argparse.ArgumentParser()
parser.add_argument('--use-stdio', action='store_true', help='Use stdio for game')
args = parser.parse_args()

def ifprint(*args_, **kwargs):
    if not args.use_stdio:
        print(*args_, **kwargs)

bot_name = 'MlrlBot'

env = halite_env.Env(stdio=args.use_stdio)
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True, bot_name=bot_name)
env = MlrlEnv(env)

actor = Illogical.make_model_actor(env)
actor.summary(print_fn=ifprint)

actor.load_weights(f'ddpg_{bot_name}_weights_final_actor.h5f')

total_reward = 0
reward = 0
observations = env.reset()
while True:
    # print(f'{observations=}')
    actions = actor.predict(observations.reshape((1, -1)))[0]

    ifprint(f'{actions=}')

    observations, reward, done, _ = env.step(actions)

    total_reward += reward
    ifprint(f'{reward=}')

    if done:
        break

env.close()
ifprint(f'done, {total_reward=}')
