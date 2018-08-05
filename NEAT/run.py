import os
import neat
import gym
import pickle
import cv2
import time

env = gym.make("MsPacman-v0")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
winner = pickle.load(open("winner.pkl", "rb"))
done = False
state = env.reset()
net = neat.nn.FeedForwardNetwork.create(winner, config)
while not done:
    state = state.reshape(33600)
    output = net.activate(state)
    ind = output.index(max(output))
    s, r, done, _ = env.step(ind)
    state = s
env.render()
time.sleep(2)
env.close()
