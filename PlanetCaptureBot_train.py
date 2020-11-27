
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TestLogger, ModelIntervalCheckpoint

import halite_env

env = halite_env.CommandEnv()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=False)
nb_actions = 28

model = Sequential()
model.add(Flatten(input_shape=(1,) + (28, 5)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000,
               target_model_update=1e-2, policy=policy, gamma=0.99)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


callbacks = [ModelIntervalCheckpoint('dqn_PlanetCaptureBot_weights_{step}.h5f', interval=100000)]

dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1, callbacks=callbacks)
dqn.save_weights('dqn_PlanetCaptureBot_weights_final.h5f', overwrite=True)

dqn.test(env, nb_episodes=1, visualize=False)
