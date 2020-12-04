
import numpy as np
import time
import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, TrainEpisodeLogger
from rl.random import OrnsteinUhlenbeckProcess

from envs import halite_env
from envs.ml_rl import MlrlEnv
from envs.metrics_env import MetricsEnv
from envs.tensorboard_callback import TensorBoard

from models import Illogical

bot_name = 'MlrlBot'
weights_name = f'ddpg_{bot_name}_weights'

parser = argparse.ArgumentParser()
parser.add_argument('--load', action='store_true', help='Continue')
args = parser.parse_args()

env = halite_env.Env()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=False, bot_name=bot_name)
env = MetricsEnv(env)
env = MlrlEnv(env)
nb_actions = env.action_space.n

actor = Illogical.make_model_actor(env)
critic, action_input = Illogical.make_model_critic(env)
actor.summary()
critic.summary()
if args.load:
    actor.load_weights(weights_name + '_final_actor.h5f')
    critic.load_weights(weights_name + '_final_critic.h5f')

# parameters


nb_steps = 20_000
nb_steps_warmup = int(nb_steps * 0.01)
memory = SequentialMemory(limit=10_000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0.0, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=nb_steps_warmup, nb_steps_warmup_actor=nb_steps_warmup,
                  random_process=random_process, gamma=0.9, target_model_update=1e-3)
agent.compile(SGD(lr=1e-5, clipvalue=0.001), metrics=['mae'])

callbacks = [
    ModelIntervalCheckpoint(weights_name + '_{step}.h5f', interval=10_000),
    TrainEpisodeLogger(),
    TensorBoard()
]

agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, callbacks=callbacks)
agent.save_weights(weights_name + '_final.h5f', overwrite=True)

# agent.test(env, nb_episodes=1, visualize=False)
