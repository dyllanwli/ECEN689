import argparse
import torch
import gym
from agents import DQNAgent, FixedDQNAgent, DoubleDQNAgent, DuelingDQNAgent
from utils.wrapper import EnvWrapper
import os


def main():
    hyper_params = {
        'memorysize': 1000000,
        'max_timesteps': 1000,
        'gamma': 0.99,
        'learning_rate': 0.005,
        'num_episodes': 1000,
        'decay_type': 'power_law',
        'batchsize': 128,
        'verbose': 4,
        'max_eps': 1.0,
        'min_eps': 0.01,
        'decay_rate': 0.999,
        'model': "DQN"}

    for key in hyper_params:
        print("{}: {}".format(key, hyper_params[key]))
    print("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solved_dir = './solved/'
    plots_dir = './plots/'

    env = gym.make("LunarLander-v2")
    seed = 18
    env_wrapper = EnvWrapper(env, device)

    agent = DQNAgent(env_wrapper.state_dim[0],
                        env_wrapper.action_dim,
                        hyper_params['learning_rate'],
                        hyper_params['gamma'],
                        hyper_params['memorysize'],
                        hyper_params['batchsize'],
                        hyper_params['max_eps'],
                        hyper_params['min_eps'],
                        hyper_params['decay_rate'],
                        device,
                        decay_type=hyper_params['decay_type'])
    if args.train:
        paths = {
            'solved_dir': solved_dir,
            'plot_dir': plots_dir
        }

        agent.train(env_wrapper,
                    paths,
                    num_episodes=hyper_params['num_episodes'],
                    max_t=hyper_params['max_timesteps'],
                    learn_every=hyper_params['learn_freq'],
                    verbose=hyper_params['verbose'])
    else:
        paths = {
            'weights': hyper_params['weights_file'],
            'plot_dir': plots_dir
        }

        agent.test(env_wrapper,
                   paths,
                   hyper_params['render'])


if __name__ == "__main__":
    main()
