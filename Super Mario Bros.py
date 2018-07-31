import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import argparse
from baselines.common import set_global_seeds
from baselines import deepq
from baselines import logger

env = gym.make("SuperMarioBros-v0")
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state = env.reset()
done = False

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='SuperMarioBros-v0')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--prioritized', type=int, default=1)
parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
parser.add_argument('--dueling', type=int, default=1)
parser.add_argument('--num-timesteps', type=int, default=int(10e6))
parser.add_argument('--checkpoint-freq', type=int, default=10)
parser.add_argument('--checkpoint-path', type=str, default="./ModelCheckpoint/")
args = parser.parse_args()
logger.configure()
set_global_seeds(args.seed)

model = deepq.models.mlp([1,1,128,1,1])

act = deepq.learn(
    env,
    q_func=model,
    lr=1e-4,
    max_timesteps=args.num_timesteps,
    buffer_size=50000,
    exploration_fraction=0.3,
    exploration_final_eps=0.01,
    print_freq=1,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=bool(args.prioritized),
    prioritized_replay_alpha=args.prioritized_replay_alpha,
    checkpoint_freq=args.checkpoint_freq,
    checkpoint_path=args.checkpoint_path,
)
print("Saving...")
act.save("mario.pkl")

"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make("RubiksCube-v0")

nb_actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)
#dqn.save_weights("./spaceinvaders.txt")
dqn.test(env, nb_episodes=5, visualize=False)"""
