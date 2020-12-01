
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, SoftmaxPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TestLogger, ModelIntervalCheckpoint, TrainEpisodeLogger

from models import LordTateKanti

import halite_env
from envs.command_env import CommandEnv

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=False)
env = CommandEnv(env)
nb_actions = env.action_space.n

model = LordTateKanti.make_model(env)
print(model.summary())

nb_steps_warmup = 1500
memory = SequentialMemory(limit=10_000, window_length=1)
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
               target_model_update=1e-2, policy=policy, gamma=0.99)
agent.compile(Adam(lr=1e-3), metrics=['mae'])


callbacks = [
    ModelIntervalCheckpoint('dqn_PlanetCaptureBot_weights_{step}.h5f', interval=100000),
    TrainEpisodeLogger()
]

agent.fit(env, nb_steps=nb_steps_warmup + 205000, visualize=False, verbose=1, callbacks=callbacks)
agent.save_weights('dqn_PlanetCaptureBot_weights_final.h5f', overwrite=True)

agent.test(env, nb_episodes=5, visualize=False)