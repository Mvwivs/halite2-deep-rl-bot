
import time
import hlt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import halite_env
from envs.command_env import CommandEnv
from models import LordTateKanti

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True)
env = CommandEnv(env)

model = LordTateKanti.make_model(env)
print(model.summary())

model.load_weights('dqn_PlanetCaptureBot_weights_final.h5f')

total_reward = 0
reward = 0
observations = env.reset()
while True:
    # print(f'{observations=}')
    actions = model.predict(observations.reshape((1, -1)))
    action = np.argmax(actions[0])

    # print(f'{actions=}')
    print(f'{action=}')

    observations, reward, done, _ = env.step(action)

    total_reward += reward
    print(f'{reward=}')

    if done:
        break

env.close()
print(f'done, {total_reward=}')
