from baselines import deepq

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gflags as flags
import sys

import numpy as np


def main(path="./models/deepq/mario_reward_1736.7.pkl"):
    step_mul = 16
    steps = 200

    FLAGS = flags.FLAGS
    flags.DEFINE_string("env", "SuperMarioBros-v0", "RL environment to train.")
    flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")

    FLAGS(sys.argv)
    # 1. Create gym environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    act = deepq.load(path)
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


if __name__ == "__main__":
    main()
