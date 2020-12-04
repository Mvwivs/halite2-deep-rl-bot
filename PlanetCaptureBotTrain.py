
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, SoftmaxPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TestLogger, ModelIntervalCheckpoint, TrainEpisodeLogger, Callback

from models import LordTateKanti

from envs import halite_env
from envs.command_env import CommandEnv
from envs.metrics_env import MetricsEnv
from envs.tensorboard_callback import TensorBoard

bot_name = 'PlanetCaptureSmartBot'

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=False, bot_name=bot_name)
env = MetricsEnv(env)
env = CommandEnv(env)
nb_actions = env.action_space.n

model = LordTateKanti.make_model(env)
model.summary()

# parameters


nb_steps = 20_000
nb_steps_warmup = int(nb_steps * 0.01)
memory = SequentialMemory(limit=10_000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=0.01, value_test=0.05, nb_steps=int(nb_steps * 0.66))
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
               target_model_update=1e-2, policy=policy, gamma=0.7)
agent.compile(Adam(lr=1e-3), metrics=['mae'])

callbacks = [
    ModelIntervalCheckpoint('dqn_PlanetCaptureBot_weights_{step}.h5f', interval=100000),
    TrainEpisodeLogger(),
    TensorBoard()
]

agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, callbacks=callbacks)
agent.save_weights('dqn_PlanetCaptureBot_weights_final.h5f', overwrite=True)

# agent.test(env, nb_episodes=1, visualize=False)
