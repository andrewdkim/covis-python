from math import exp
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class Procedural:
    def __init__(
        self,
        category_a=1,
        category_b=2,
        base_dopamine = 0.20,
        alpha = 95,
        alpha_w = 0.016, #0.016,
        beta_w = 0.0035, #0.0035,
        gamma_w = 0.0006,
        theta_nmda = 0.0022,
        theta_ampa = 0.01,
        w_max = 1,
        sigma_p = 0.0125,

    ) -> None:
        self.actual_category = None
        self.stimulus = []
        self.base_dopamine = base_dopamine
        self.alpha = alpha
        self.alpha_w = alpha_w
        self.beta_w = beta_w
        self.gamma_w = gamma_w
        # self.w_inhib = w_inhib

        self.theta_nmda = theta_nmda
        self.theta_ampa = theta_ampa
        self.w_max = w_max
        self.sigma_p = sigma_p

        self.category_a = category_a
        self.category_b = category_b

        self.num_striatal = 2
        self.num_sensory = 10000
        self.sensory_coordinates = list(product(range(1, 101), range(1, 101)))

        self.prev_predicted_reward = 0  # initially, p_0 = 0
        self.prev_obtained_reward = 0
        self.weights = self.initial_weight()
        self.predicted_category = None
        self.max_confidence = -np.inf

    def obtained_reward(self):
        actual_category = self.actual_category
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
        if rpe > 1:
            return 1
        elif -0.25 < rpe and rpe <= 1:
            return 0.8 * rpe + self.base_dopamine
        return 0

    def sensory_activation(self, k):
        row, col = self.sensory_coordinates[k - 1] # col must match stri
        sti1, sti2 = self.stimulus[0], self.stimulus[1] #sti1 = stri
        dist = np.linalg.norm(np.array([sti1, sti2]) - np.array([col, 100 - row]))
        return exp(-(dist ** 2) / self.alpha)

    def striatal_activation(self, j):
        striatal_sum = 0
        for k in range(1, self.num_sensory + 1):
            w_kj = self.weights[k - 1][j - 1]
            striatal_sum += ((w_kj * self.sensory_activation(k))) 
        # print(str(j) + ": " + str(striatal_sum))
        return striatal_sum + np.random.normal(0, self.sigma_p)

    def initial_weight(self):
        def init_weight_fn(): return 0.001 + 0.0025 * random.uniform(0, 1)
        weights = np.zeros((self.num_sensory, self.num_striatal))
        for i in range(self.num_sensory):
            for j in range(self.num_striatal):
                weights[i][j] = init_weight_fn()
        return weights

    def next_weight(self, curr_weight, s_a, s_b, dopamine):
        new_weights = np.copy(curr_weight)
        striatal_activations = [s_a, s_b]
        d_n = dopamine
        predicted_category_index = self.predicted_category - 1
        for k in range(1, self.num_sensory + 1):  # 1, 2, ... 10000
            i_k = self.sensory_activation(k)
            s_j = striatal_activations[predicted_category_index] #inhibit other category from updates
            curr_w_kj = curr_weight[k - 1][predicted_category_index]
            first_term = self.alpha_w * i_k * \
                max(s_j - self.theta_nmda, 0) * max(d_n -
                                                    self.base_dopamine, 0) * (self.w_max - curr_w_kj)
            second_term = self.beta_w * i_k * \
                max(s_j - self.theta_nmda, 0) * \
                max(self.base_dopamine - d_n, 0) * curr_w_kj
            third_term = self.gamma_w * i_k * \
                max(max(self.theta_nmda - s_j, 0) -
                    self.theta_ampa, 0) * curr_w_kj
            new_weight = curr_w_kj + first_term - second_term - third_term
            if new_weight <= 0:
                new_weight = 0
            elif new_weight >= 1:
                new_weight = 1
            new_weights[k - 1][predicted_category_index] = new_weight
        return new_weights

    def make_decision(self, s_a, s_b):
        return self.category_a if s_a > s_b else self.category_b

    def graph_sensory(self, trial):
        """
        Just for debugging / testing - visualizes sensory model
        """
        print("Category: " + "A" if trial[0] == 1 else "B")
        print("Striatal: " + str(trial[1]))
        print("Orientation: " + str(trial[2]))
        temp = []
        for k in range(1, self.num_sensory + 1):  # 1, 2, ... 10000
            temp.append(self.sensory_activation(k))
        sensory_plot = np.array(temp).reshape((100, 100))
        plt.imshow(sensory_plot, cmap='viridis')
        plt.colorbar()
        plt.show()

    def run_trial(self, trial):
        self.actual_category = trial[0]
        self.stimulus = trial[1:]
        # self.graph_sensory(trial)
        s_a = self.striatal_activation(self.category_a)
        s_b = self.striatal_activation(self.category_b)
        # print( "s_a: " + str(s_a))
        # print( "s_b: " + str(s_b))
        self.predicted_category = self.make_decision(s_a, s_b)
        obtained_reward = self.obtained_reward()
        predicted_reward = self.predicted_reward()
        dopamine = self.dopamine(obtained_reward, predicted_reward)
        curr_weights = self.weights.copy()
        self.weights = self.next_weight(self.weights.copy(), s_a, s_b, dopamine).copy()
        self.prev_obtained_reward = obtained_reward
        self.prev_predicted_reward = predicted_reward
        confidence = abs(s_a - s_b)
        self.max_confidence = max(confidence, self.max_confidence)
        return self.predicted_category, confidence, curr_weights
