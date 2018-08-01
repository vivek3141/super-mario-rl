import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import argparse
from baselines.common import set_global_seeds
from baselines import deepq
from baselines import logger
from ActWrapper import ActWrapper
import os

max_mean_reward = 0
last_filename = ""


def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SuperMarioBros-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=bool, default=False)
    parser.add_argument('--num-timesteps', type=int, default=2000000)
    parser.add_argument('--checkpoint-freq', type=int, default=10)
    parser.add_argument('--checkpoint-path', type=str, default="./ModelCheckpoint/")
    args = parser.parse_args()
    return args


def callback(locals, globals):
    PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
    global max_mean_reward, last_filename
    if 'done' in locals and locals['done'] == True:
        if ('mean_100ep_reward' in locals
                and locals['num_episodes'] >= 10
                and locals['mean_100ep_reward'] > max_mean_reward
        ):
            print("mean_100ep_reward : %s max_mean_reward : %s" %
                  (locals['mean_100ep_reward'], max_mean_reward))

            if not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq/')):
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/'))
                except Exception as e:
                    print(str(e))
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/deepq/'))
                except Exception as e:
                    print(str(e))

            if last_filename != "":
                os.remove(last_filename)
                print("delete last model file : %s" % last_filename)

            max_mean_reward = locals['mean_100ep_reward']
            act = ActWrapper(locals['act'], locals['act_params'])

            filename = os.path.join(PROJ_DIR, 'models/deepq/mario_reward_%s.pkl' % locals['mean_100ep_reward'])
            act.save(filename)
            print("save best mean_100ep_reward model to %s" % filename)
            last_filename = filename


args = parse()
logger.configure()
set_global_seeds(args.seed)
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
state = env.reset()
done = False

model = deepq.models.cnn_to_mlp(
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=args.dueling
)

act = deepq.learn(
    env,
    q_func=model,
    lr=5e-4,
    max_timesteps=args.num_timesteps,
    buffer_size=10000,
    exploration_fraction=0.5,
    exploration_final_eps=0.01,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=False,
    callback=callback,
    print_freq=1
)
act.save("mario_model.pkl")
env.close()
