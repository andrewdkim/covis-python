from math import exp
import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from itertools import product


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
        self.sensory_coordinates = list(product(range(1, 101), range(1, 101)))

        self.prev_predicted_reward = 0  # initially, p_0 = 0
        self.prev_obtained_reward = 0
        self.weights = self.initial_weight()
        self.predicted_category = None

        self.output = []

    def rescale_trials_RB_triangle(self):
        orientation = self.trials[:, 1] # "x"
        spatial = self.trials[:, 2] # "y"
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
        self.trials = np.column_stack((self.trials[:, 0], orientation, spatial))

    def curr_trial(self):
        return self.trials[self.n - 1]
    
    def curr_trial_category(self):
        return int(self.curr_trial()[0])

    def obtained_reward(self):
        actual_category = self.curr_trial_category()
        predicted_category = self.predicted_category
        if actual_category == predicted_category:
            return 1
        elif actual_category != predicted_category:
            return -1
        else:
            return 0

    def predicted_reward(self):
        return self.prev_predicted_reward + 0.025 * (self.prev_obtained_reward - self.prev_predicted_reward)

    def dopamine(self, obtained_reward, predicted_reward):
        rpe = obtained_reward - predicted_reward
        print("rpe: " + str(rpe))
        if rpe > 1:
            return 1
        elif -0.25 < rpe and rpe <= 1:
            return 0.8 * rpe + self.base_dopamine
        return 0

    def sensory_activation(self, k):
        row, col = self.sensory_coordinates[k - 1]
        sti1, sti2 = self.curr_trial()[1], self.curr_trial()[2]
        dist = np.linalg.norm(np.array([sti1, sti2]) - np.array([row, col]))
        return exp(-(dist ** 2) / self.alpha)

    def striatal_activation(self, j):
        striatal_sum = 0
        for k in range(1, self.num_sensory + 1):
            w_kj = self.weights[k - 1][j - 1]
            striatal_sum += (w_kj * self.sensory_activation(k) + np.random.normal(0, self.sigma_p))
        return striatal_sum

    def initial_weight(self):
        def init_weight_fn(): return 0.001 + 0.0025 * random.uniform(0, 1)
        weights = np.zeros((self.num_sensory, self.num_striatal))
        for i in range(self.num_sensory):
            for j in range(self.num_striatal):
                weights[i][j] = init_weight_fn()
        return weights

    def next_weight(self, curr_weight, s_a, s_b, dopamine):
        new_weights = np.zeros((self.num_sensory, self.num_striatal))
        striatal_activations = [s_a, s_b]
        d_n = dopamine
        for k in range(1, self.num_sensory + 1): # 1, 2, ... 10000
            i_k = self.sensory_activation(k)
            for j in range(1, self.num_striatal + 1): # 1, 2
                s_j = striatal_activations[j - 1]
                curr_w_kj = curr_weight[k - 1][j - 1]
                first_term = self.alpha_w * i_k * max(s_j - self.theta_nmda, 0) * max(d_n - self.base_dopamine, 0) * (self.w_max - curr_w_kj)
                second_term = self.beta_w * i_k * max(s_j - self.theta_nmda, 0) * max(self.base_dopamine - d_n, 0) * curr_w_kj
                third_term = self.gamma_w * i_k * max(max(self.theta_nmda - s_j, 0) - self.theta_ampa, 0) * curr_w_kj

                new_weight = curr_w_kj + first_term - second_term - third_term
                if new_weight <= 0:
                    new_weight = 0
                elif new_weight >= 1:
                    new_weight = 1
                new_weights[k - 1][j - 1] = new_weight
        return new_weights

    def make_decision(self, s_a, s_b):
        return self.category_a if s_a > s_b else self.category_b

    def run_trials(self):
        for i, _ in enumerate(self.trials):
            if i % 100 == 0:
                print(str(i) + "th iteration")
                self.generate_heatmap(self.weights[:, 0].reshape((100, 100)))
            s_a = self.striatal_activation(self.category_a)
            s_b = self.striatal_activation(self.category_b)
            # self.predicted_category = self.make_decision(s_a, s_b)
            self.predicted_category = self.curr_trial_category() #TEST - always correct
            obtained_reward = self.obtained_reward()
            predicted_reward = self.predicted_reward()
            dopamine = self.dopamine(obtained_reward, predicted_reward)
            print("dopa: " + str(dopamine))

            #prepare for next iteration
            self.weights = self.next_weight(self.weights.copy(), s_a, s_b, dopamine).copy()
            self.prev_obtained_reward = obtained_reward
            self.prev_predicted_reward = predicted_reward
            # print(self.prev_obtained_reward)
            # print(self.prev_predicted_reward)
            self.n += 1
            self.output.append([self.predicted_category])
    
    def generate_heatmap(self, arr):
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.show()
    
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

    