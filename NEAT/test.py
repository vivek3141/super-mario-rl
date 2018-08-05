import cv2
import gym_super_mario_bros as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
env = gym.make("SuperMarioBros-v3")
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state = env.reset()
state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
cv2.imshow('a', state)
cv2.waitKey(0)
cv2.destroyAllWindows()