
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.stderr = open('errors.log', 'w')

import time
import hlt
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from envs import halite_env
from envs.command_env import CommandEnv
from models import LordTateKanti

parser = argparse.ArgumentParser()
parser.add_argument('--use-stdio', action='store_true', help='Use stdio for game')
args = parser.parse_args()

def ifprint(*args_, **kwargs):
    if not args.use_stdio:
        print(*args_, **kwargs)

bot_name = 'PlanetCaptureBot'

env = halite_env.Env(stdio=args.use_stdio)
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True, bot_name=bot_name)
env = CommandEnv(env)

model = LordTateKanti.make_model(env)
model.summary(print_fn=ifprint)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_location = os.path.join(current_directory, 'dqn_PlanetCaptureBot_weights_final.h5f')
model.load_weights(model_location)

total_reward = 0
reward = 0
observations = env.reset()
while True:
    # print(f'{observations=}')
    actions = model.predict(observations.reshape((1, -1)))
    action = np.argmax(actions[0])

    ifprint(f'{actions=}')

    observations, reward, done, _ = env.step(action)

    total_reward += reward
    ifprint(f'{reward=}')

    if done:
        break

env.close()
ifprint(f'done, {total_reward=}')
