from baselines import deepq

import gym

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gflags as flags
import sys

import numpy as np

import baselines.common.tf_util as U

step_mul = 16
steps = 200

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_string("file", "model_bak.pkl", "Trained model file to use.")

FLAGS(sys.argv)
# 1. Create gym environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

act = deepq.load("models/deepq/%s" % FLAGS.file)
nstack = 4
nh, nw, nc = env.observation_space.shape
history = np.zeros((1, nh, nw, nc * nstack), dtype=np.uint8)

obs, done = env.reset(), False
# history = update_history(history, obs)
episode_rew = 0
while not done:
    env.render()
    action = act([obs])[0]
    obs, rew, done, _ = env.step(action)
    # history = update_history(history, obs)
    episode_rew += rew
    print("action : %s reward : %s" % (action, rew))

print("Episode reward", episode_rew)
