import numpy as np
import gym


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy,EpsGreedyQPolicy
from rl.memory import SequentialMemory
import os

ENV_NAME = 'LunarLander-v2'
_path = f'./save_model/LunarLander/NoPiven'
if not os.path.exists(_path):
    os.makedirs(_path)
seed=1235
# Get the environment and extract the number of actions.
# env = DescrteToContWrapper(gym.make(ENV_NAME))
while(True):
    env =gym.make(ENV_NAME)

    np.random.seed(seed)
    env.seed(seed)
    nb_actions = 4

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy,enable_double_dqn=True)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights(f'{_path}/dqn_{ENV_NAME}_weights_seed{seed}.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)
    seed+=1