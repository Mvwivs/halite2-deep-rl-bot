import time

import numpy as np

import halite_env
from envs.superior_env import SuperiorEnv
from models import LordTateKanti

env = SuperiorEnv(
    env=halite_env.Env(),
    tiles_num=16,
)
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True)

model = LordTateKanti.make_model(env)
print(model.summary())

model.load_weights('dqn_supbot_bk.h5f')

reward = 0
observations = env.reset()
while True:
    action = np.argmax(model.predict(observations.reshape((1, -1))))

    # print(f'{actions=}')
    print(f'{action=}')

    observations, reward, done, _ = env.step(action)

    print(f'{reward=}')

    if done:
        break

env.close()
print("done")
