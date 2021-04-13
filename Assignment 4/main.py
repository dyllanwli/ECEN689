import argparse
import gym
import os
import sys
import pickle
import time

from utils import *
from torch import nn
from agent import Agent

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy

from point_env import PointEnv
from point_mass_solutions import estimate_net_grad

parser = argparse.ArgumentParser(description='Pytorch Policy Gradient')
parser.add_argument('--env-name', default="Point-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=True,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
if args.env_name == 'CartPole-v0':
    running_state = ZFilter((state_dim,), clip=5)


"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""cuda setting"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)


"""define actor and critic"""
if args.env_name == 'Point-v0':
    # we use only a linear policy for this environment
    theta = torch.normal(0, 0.01, size=(action_dim, state_dim + 1))
    policy_net = None
    theta = theta.to(dtype).to(device)
else:
    # we use both a policy and a critic network for this environment
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
    theta = None
    value_net = Value(state_dim)
    to_device(device, policy_net, value_net)

    # Optimizers
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)


"""create agent"""
if args.env_name == 'Point-v0':
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
              running_state=None, num_threads=args.num_threads)
else:
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                  running_state=running_state, num_threads=args.num_threads)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

    if args.env_name == 'CartPole-v0':
        with torch.no_grad():
            values = value_net(states)

    if args.env_name == 'Point-v0':
        """get values estimates from the trajectories"""
        net_grad = estimate_net_grad(rewards, masks, values, args.gamma, args.tau, device)

        """update policy parameters"""
        theta += args.learning_rate * net_grad



def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_reward'], log_eval['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            # you will have to specify a proper directory to save files
            pickle.dump((policy_net, value_net), open(os.path.join(directory_for_saving_files, 'learned_models/{}_policy_grads.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

main_loop()