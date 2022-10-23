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
        self.alpha = 10
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
        self.predicted_category = None
        self.prev_striatal_activations = []
        self.prev_sensory_activations = []

        self.output = []

    def rescale_trials_RB_triangle(self):
        orientation = self.trials[:, 1] # "x"
        spatial = self.trials[:, 2] # "y"
        plt.scatter(orientation, spatial)
        plt.show()
        min_orientation = abs(min(orientation))
        min_spatial = abs(min(spatial))
        padding = 5
        orientation = [x + min_orientation for x in orientation]
        spatial = [y + min_spatial for y in spatial]
        plane = np.column_stack((orientation, spatial))
        plane *= (100 - 2 * padding) / plane.max()
        orientation = plane[:, 0]
        spatial = plane[:, 1]
        orientation = [x + padding for x in orientation]
        spatial = [y + padding for y in spatial]
        plt.scatter(orientation, spatial)
        plt.show()
        self.trials = np.column_stack((self.trials[:, 0], orientation, spatial))

    def curr_trial(self):
        return self.trials[self.n - 1]

    def obtained_reward(self):
        actual_category = int(self.curr_trial()[0])
        predicted_category = self.predicted_category
        if actual_category == predicted_category:
            return 1
        elif actual_category != predicted_category:
            return -1
        else:
            return 0

    def predicted_reward(self):
        return self.prev_predicted_reward + 0.025 * (self.prev_obtained_reward - self.prev_predicted_reward)

    def dopamine(self):
        rpe = self.obtained_reward() + self.predicted_reward()
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
        return exp(-(dist ** 2) / self.alpha)

    def striatal_activation(self, j):
        striatal_sum = 0
        for k in range(1, self.num_sensory):
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

    def next_weight(self, curr_weight, s_a, s_b):
        weights = np.zeros((self.num_sensory, self.num_striatal))
        striatal_activations = [s_a, s_b]
        d_n = self.dopamine()
        for k in range(1, self.num_sensory):
            i_k = self.sensory_activation(k)
            for j, s_j in enumerate(striatal_activations):
                weights[k - 1][j - 1] = curr_weight[k - 1][j - 1]
                + self.alpha_w * i_k * max((s_j - self.theta_nmda), 0) * max(d_n - self.base_dopamine, 0) * (self.w_max - curr_weight[k - 1][j - 1])
                - self.beta_w * i_k * max(s_j - self.theta_nmda, 0) * max(self.base_dopamine - d_n, 0) * curr_weight[k - 1][j - 1]
                - self.gamma_w * i_k * max(max(self.theta_nmda - s_j, 0) - self.theta_ampa, 0) * curr_weight[k - 1][j - 1]
        return weights

    def make_decision(self, s_a, s_b):
        return self.category_a if s_a > s_b else self.category_b

    def run_trials(self):
        for i, _ in enumerate(self.trials):
            s_a = self.striatal_activation(self.category_a)
            s_b = self.striatal_activation(self.category_b)
            self.predicted_category = self.make_decision(s_a, s_b)
            self.weights = self.next_weight(self.weights, s_a, s_b)
            self.n += 1
            self.output.append([self.predicted_category])
    
    
    def generate_output(self, txt_file_path, batch_size=50):
        #saving output to txt file
        output = np.append(self.trials, self.output, 1)
        np.savetxt(txt_file_path, output, fmt='%1.3f')

        #graphing learning rate
        num_batch = int(len(output) / batch_size)
        x = []
        y = []
        for i in range(num_batch):
            batch = output[i : i + batch_size]
            actual_categories = batch[:, 0]
            predicted_categories= batch[:, 3]
        
            num_correct = 0
            for j,  predicted_category in enumerate(predicted_categories):
                actual_category = actual_categories[j]
                if predicted_category == actual_category:
                    num_correct += 1
            x.append(i)
            y.append(num_correct / batch_size)

        plt.plot(x, y)
        plt.xlabel("Batch")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label
        plt.title("Learning Curve of Each Batch and its Accuracy")  # add title
        plt.xticks(np.arange(len(x)))
        plt.show()

    