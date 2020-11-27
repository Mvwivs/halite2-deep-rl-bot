
import time
import hlt
import halite_env
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

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

model.load_weights('dqn_PlanetCaptureBot_weights_final.h5f')

env = halite_env.CommandEnv()
env.configure(socket_path=f"/dev/shm/{time.time_ns()}", replay=True)

reward = 0
observations = env.reset()
while True:
    # print(f'{observations=}')

    actions = model.predict(observations.reshape((-1, 140)))
    action = np.argmax(actions[0])

    # print(f'{actions=}')
    print(f'{action=}')

    observations, reward, done, _ = env.step(action)

    print(f'{reward=}')

    if done:
        break

env.close()
print("done")
