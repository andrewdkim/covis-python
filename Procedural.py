from math import exp
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
from random import choice


class Procedural:
    def __init__(self, trials, category_a=1, category_b=2) -> None:
        self.trials = trials
        self.base_dopamine = 0.20
        self.alpha = 50
        self.alpha_w = 0.65
        self.beta_w = 0.19
        self.gamma_w = 0.02
        self.theta_nmda = 0.0022
        self.theta_ampa = 0.01
        self.w_max = 1
        self.sigma_p = 0.0125

        self.n = 1  # current trial

        self.category_a = category_a
        self.category_b = category_b

        self.num_striatal = 2
        self.num_sensory = 10000

        self.prev_predicted_reward = 0  # initially, p_0 = 0
        self.prev_obtained_reward = 0
        self.weights = self.initial_weight()
        self.prev_striatal_activations = []
        self.prev_sensory_activations = []

    def calc_obtained_reward(self):
        pass

    def calc_predicted_reward(self):
        return self.prev_predicted_reward + 0.025 * (self.prev_obtained_reward - self.prev_predicted_reward)

    def curr_trial(self):
        return self.trials[self.n - 1]

    def dopamine(self):
        rpe = self.calc_obtained_reward() + self.calc_predicted_reward()
        if rpe > 1:
            return 1
        elif -0.25 < rpe and rpe <= 1:
            return 0.8 * rpe + self.base_dopamine
        return 0

    def sensory_activation(self, k):
        row = int(k / 100) + 1
        col = int(k % 100)
        if k % 100 == 0:
            row = row - 1
            col = 100
        sti1, sti2 = self.curr_trial()[1], self.curr_trial()[2]
        dist = np.linalg.norm(np.array([sti1, sti2]) - np.array([row, col]))
        return exp((dist ** 2) / self.alpha)

    # j >= 1

    def striatal_activation(self, j):
        striatal_sum = 0
        for k in range(1, self.num_sensory):
            print(k)
            striatal_sum += self.weights[k - 1][j - 1] * self.sensory_activation(
                k) + np.random.normal(0, self.sigma_p ** 2)
        return striatal_sum

    def initial_weight(self):
        def init_weight_fn(): return 0.001 + 0.0025 * random.uniform(0, 1)
        weights = np.zeros((self.num_sensory, self.num_striatal))
        for i in range(self.num_sensory):
            for j in range(self.num_striatal):
                weights[i][j] = init_weight_fn()
        return weights

    def next_weight(self, curr_weight):
        weights = np.zeros((self.num_sensory, self.num_striatal))
        for k in range(1, self.num_sensory):
            for j in range(1, self.num_striatal):
                weights[k - 1][j - 1] = curr_weight[k - 1][j - 1]
                + self.alpha_w * self.sensory_activation(k) * max((self.striatal_activation(j) - self.theta_nmda), 0) * max(
                    self.dopamine() - self.base_dopamine, 0) * (self.w_max - curr_weight[k - 1][j - 1])
                - self.beta_w * self.sensory_activation(k) * max(self.striatal_activation(
                    j) - self.theta_nmda, 0) * max(self.base_dopamine - self.dopamine()) * curr_weight[k - 1][j - 1]
                - self.gamma_w * self.sensory_activation(k) * max(max(
                    self.theta_nmda - self.striatal_activation(j), 0) - self.theta_ampa, 0) * curr_weight[k - 1][j - 1]
        return weights

    def make_decision(self):
        return self.category_a if self.striatal_activation(self.category_a) > self.striatal_activation(self.category_b) else self.category_b

    def run_trials(self):
        print(self.striatal_activation(1))
