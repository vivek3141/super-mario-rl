import os
import neat
import gym, gym_super_mario_bros
import pickle
import cv2

env = gym_super_mario_bros.make("SuperMarioBros-v3")


def eval_genomes(genomes, config):
    for g_id, genome in genomes:
        done = False
        state = env.reset()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        while not done:
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = state.reshape(61440)
            # env.render()
            output = net.activate(state)
            ind = output.index(max(output))
            s, reward, done, _ = env.step(ind)
            state = s
            genome.fitness += reward


def run(config_file):
    print("Step")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    print("Step")
    p = neat.Population(config)
    print("Step")
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 3)
    pickle.dump(winner, open('winner.pkl', 'wb'))
    print("Done!")


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
