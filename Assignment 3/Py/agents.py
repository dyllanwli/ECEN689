from abc import abstractmethod

import torch.optim as optim
import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.common import Experience
from utils.replay import ReplayMemory
from models import DQN


class Agent:
    def __init__(self,
                 experiences_buffer,
                 batch_size,
                 max_eplison,
                 min_eplison,
                 decays,
                 device,
                 decay_type="power_law"):

        self.model_name = "Base"

        def power_law_decay(curr_eps):
            return max(
                min_eplison, curr_eps * decays
            )

        decay_options = {
            "power_law": power_law_decay,
        }

        if decay_type not in decay_options:
            raise ValueError("decay_type parameter not valid. Expected one of {}"
                             .format([i for i in decay_options.keys()]))

        self.decay_law = decay_options[decay_type]
        self.decay_type = decay_type

        self.device = device

        # policy
        self.max_eplison = max_eplison
        self.min_eplison = min_eplison
        self.decays = decays
        self.curr_step = 0
        self.curr_eps = max_eplison

        # replay memory
        self.experiences = ReplayMemory(experiences_buffer, device)
        self.batch_size = batch_size

    def update_hyparameter(self):
        if self.decay_type == 'exp':
            self.curr_eps = self.decay_law(self.curr_step)
        else:
            self.curr_eps = self.decay_law(self.curr_eps)

        return self.curr_eps

    def append_to_experiences(self, s, a, r, s_, d):
        self.experiences.append(Experience(s, a, r, s_, d))

    def check_experience(self):
        return self.experiences.can_sample(self.batch_size)

    @staticmethod
    def plot(scores, avg_period, winning_score, eps=None, filename=None):
        def moving_avg():
            avg = []
            for i in range(len(scores)):
                if i < avg_period:
                    avg.append(0)
                else:
                    avg.append(np.mean(scores[i - avg_period:i]))

            return avg

        fig, axis = plt.subplots()
        axis.clear()
        axis.plot(scores, 'blue', label='Score per episode', alpha=0.4)
        axis.plot(moving_avg(), 'black',
                  label='Mean score of the last {} episodes'.format(avg_period))
        axis.axhline(winning_score, c='green', label='Winning score', alpha=0.7)
        axis.axhline(0, c='grey', ls='--', alpha=0.7)
        axis.set_xlabel('Episodes')
        axis.set_ylabel('Scores')
        axis.legend()

        if eps is not None:
            tw_axis = axis.twinx()
            tw_axis.plot(eps, 'red', alpha=0.5)
            tw_axis.set_ylabel('Epsilon', color='red')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close()

    def train(self,
              env,
              paths,
              num_episodes=2000,
              max_t=1000,
              learn_every=4,
              verbose=2,
              avg_period=100,
              winning_score=200,
              plot_freq=200,
              use_exp=True):
        epsilons, losses, scores = [], [], []
        updates = 0
        last_mean = 0
        avg = []

        self.set_train()
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        for ep in range(1, num_episodes + 1):
            state = env.reset()
            score = 0
            for t in range(1, max_t + 1):
                if verbose == 3:
                    env.render()

                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                if use_exp:
                    self.append_to_experiences(state, action, reward, next_state, done)
                    score += reward
                    state = next_state

                    if self.check_experience() and self.curr_step % learn_every == 0:
                        loss = self.learn()
                        losses.append(loss)
                        updates += 1

                    if done:
                        self.update_hyparameter()
                        break
                else:
                    
                    learn_without_exp(state, action, reward, next_state, done)
                    score += reward
                    state = next_state
                    if done:
                        self.update_hyparameter()

            scores.append(score)
            epsilons.append(self.curr_eps)

            if ep % plot_freq == 0 and verbose > 0:
                # plotting to see behaviour: not saving to file
                self.plot(scores,
                          avg_period,
                          winning_score,
                          epsilons)

            avg_reward = np.mean(scores[-avg_period:])
            avg.append(avg_reward)
            if avg_reward > winning_score and avg_reward > last_mean:
                last_mean = avg_reward
                self.plot(scores,
                          avg_period,
                          winning_score,
                          epsilons,
                          filename=paths['plot_dir'] + self.model_name + '_train_' + curr_time + '.png')

        if verbose > 0:
            print("Training finished.")
            self.plot(scores,
                      avg_period,
                      winning_score,
                      epsilons)
        env.close()

    def test(self,
             env,
             paths,
             render=True,
             num_episodes=100,
             max_t=1000,
             winning_score=200):
        try:
            self.load(paths['weights'])
        except FileNotFoundError:
            print("File not found.")
            exit(1)

        self.set_test()

        test_scores = []
        for episode in range(1, num_episodes + 1):
            s = env.reset()
            score = 0
            for t in range(1, max_t + 1):
                if render:
                    env.render()

                a = self.choose_action(s, testing=True)
                s_, r, d = env.step(a)
                score += r
                s = s_
                t += 1
                if d:
                    break

            test_scores.append(score)

            print("Episode {} - score {}\n".format(episode, score))

        if paths['plot_dir'] is not None:
            plt.axhline(winning_score, c='green', label='Winning score', alpha=0.7)
            plt.plot(test_scores, c='blue', label='Score per episode')
            curr_time = datetime.now().strftime("%Y%m%d_%H%M")
            plt.savefig(paths['plot_dir'] + self.model_name + '_test_' + curr_time + '.png')
        print("Testing finished.")
        test_scores = np.array(test_scores)
        success = test_scores[test_scores >= 200]
        print("Success rate: {}% - highest score: {}"
              .format(len(success) * num_episodes / 100, np.max(test_scores)))

        env.close()


class DQNAgent(Agent):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lr,
                 gamma,
                 experiences_buffer,
                 batch_size,
                 max_eplison,
                 min_eplison,
                 decays,
                 device,
                 linear1_units=64,
                 linear2_units=64,
                 decay_type="linear"):

        super().__init__(experiences_buffer,
                         batch_size,
                         max_eplison,
                         min_eplison,
                         decays,
                         device,
                         decay_type)

        self.model_name = "DQN"
        self.output_dim = output_dim
        self.policy_net = DQN(input_dim, output_dim, linear1_units, linear2_units).to(device)

        # optimizer
        self.optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def choose_action(self, state, testing=False):

        self.curr_step += 1

        if not testing and np.random.random() < self.curr_eps:
            return np.random.randint(0, self.output_dim)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def learn(self):

        states, next_states, actions, rewards, dones = self.experiences.sample(self.batch_size)

        q = self.policy_net(states).gather(1, actions)
        q_ = self.policy_net(next_states).max(1, keepdim=True)[0].detach()
        optimal_q = (rewards + self.gamma * q_ * (1 - dones)).to(self.device)
        loss = F.smooth_l1_loss(q, optimal_q)
        self.optim.zero_grad()
        loss.backward()

        self.optim.step()

        return loss.item()
    
    def learn_without_exp(self, state, action, reward, next_state, done):

        q = self.policy_net(states).gather(1, actions)
        q_ = self.policy_net(next_states).max(1, keepdim=True)[0].detach()

        optimal_q = (reward + self.gamma * q_ * (1 - done)).to(self.device)
        # expected q value
        loss = F.smooth_l1_loss(q, optimal_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def set_test(self):
        self.policy_net.eval()

    def set_train(self):
        self.policy_net.train()

    def save(self, filename):

        self.policy_net.save(filename)

    def load(self, filename):

        self.policy_net.load(filename, self.device)
