import random
import numpy as np
import matplotlib.pyplot as plt
from random import choice



class Procedural:
    def __init__(self, trials, rules, delta_c = 0.0025, delta_e = 0.02, gamma = 1, lam = 5, category_a = 1, category_b = 2) -> None:
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

        self.prev_predicted_reward = 0 #initially, p_0 = 0
        self.prev_obtained_reward = 0
        self.n = 1  # current trial


    def calc_obtained_reward(self):
        pass

    def calc_predicted_reward(self):
        return self.prev_predicted_reward + 0.025 * (self.prev_obtained_reward - self.prev_predicted_reward)


    def calc_dopamine(self):
        rpe = self.calc_obtained_reward() + self.calc_predicted_reward()
        if rpe > 1:
            return 1
        elif -0.25 < rpe and rpe <= 1:
            return 0.8 * rpe + self.base_dopamine
        return 0

    def calc_base_weight(self):
        return 0.001 + 0.0025 * random.uniform(0, 1)

    
    def calc_sensory_activation(self, k):
        row = int(k / 10)
        col = int(k % 10)


    

    