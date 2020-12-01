import time

from rl.agents.dqn import DQNAgent
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy
from tensorflow.keras.optimizers import Adam

import halite_env
from envs.superior_env import SuperiorEnv
from models import LordTateKanti

env = SuperiorEnv(
    env=halite_env.Env(),
    tiles_num=16,
)
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=False)
nb_actions = env.action_space.n

model = LordTateKanti.make_model(env)
print(model.summary())

memory = SequentialMemory(limit=10_000, window_length=1)
policy = BoltzmannGumbelQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy, gamma=0.99)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

callbacks = [
    #ModelIntervalCheckpoint('dqn_PlanetCaptureBot_weights_{step}.h5f', interval=100),
    TrainEpisodeLogger()
]

dqn.fit(env, nb_steps=100_000, visualize=False, verbose=0, callbacks=callbacks)
dqn.save_weights('dqn_SuperiorBot_weights_final.h5f', overwrite=True)

