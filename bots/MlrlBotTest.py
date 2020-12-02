
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import time
import hlt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import halite_env
from envs.ml_rl import MlrlEnv
from models import Illogical

bot_name = 'MlrlBot'

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True, bot_name=bot_name)
env = MlrlEnv(env)

actor = Illogical.make_model_actor(env)
print(actor.summary())

actor.load_weights(f'ddpg_{bot_name}_weights_final_actor.h5f')

total_reward = 0
reward = 0
observations = env.reset()
while True:
    # print(f'{observations=}')
    actions = actor.predict(observations.reshape((1, -1)))[0]

    print(f'{actions=}')

    observations, reward, done, _ = env.step(actions)

    total_reward += reward
    print(f'{reward=}')

    if done:
        break

env.close()
print(f'done, {total_reward=}')
