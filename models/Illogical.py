
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Concatenate, Input

def make_model_actor(env):
    nb_actions = env.action_space.n
    init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(500))
    actor.add(Activation('sigmoid'))
    actor.add(Dense(300))
    actor.add(Activation('sigmoid'))
    actor.add(Dense(100))
    actor.add(Activation('sigmoid'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid'))

    return actor

def make_model_critic(env):
    nb_actions = env.action_space.n

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)

    x = Dense(500)(flattened_observation)
    x = Activation('sigmoid')(x)
    x = Concatenate()([x, action_input])
    x = Dense(300)(x)
    x = Activation('sigmoid')(x)
    x = Dense(100)(x)
    x = Activation('sigmoid')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)

    return critic, action_input
