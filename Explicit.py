import numpy as np
import matplotlib.pyplot as plt
from random import choice



class Explicit:
    def __init__(self, rules, delta_c = 0.0025, delta_e = 0.02, gamma = 1, lam = 5, category_a = 1, category_b = 2) -> None:
        self.rules = rules
        self.prev_trial = None
        self.trial = None
        self.delta_c = delta_c
        self.delta_e = delta_e
        self.gamma = gamma
        self.lam = lam
        self.category_a = category_a
        self.category_b = category_b

        self.prev_rule_index = None
        # when starting out, rule Ri is randomly chosen
        self.current_rule_index = np.random.randint(0, len(rules))
        self.prev_prediction = None
        # initially, all salience is set to 0.25, assuming binary rules
        self.saliences = np.full(len(self.rules), 0.25)
        self.weights = np.zeros(len(self.rules))
        self.output = []
        self.output_saliences = []

    def poisson_dist(self):
        return np.random.poisson(lam=self.lam)

    def predict(self):
        return self.rules[self.current_rule_index](self.trial)

    # should only be called in n => 2

    def update_salience(self):
        prev_category = int(self.prev_trial[0])
        if (self.prev_prediction == prev_category):
            self.saliences[self.prev_rule_index] = self.saliences[self.prev_rule_index] + self.delta_c
        else:
            self.saliences[self.prev_rule_index] = self.saliences[self.prev_rule_index] - self.delta_e
    
    def run_trial(self, trial):
        if self.prev_rule_index != None:
            self.update_salience()
        self.trial = trial
        actual_category = int(trial[0])
        predicted_category = self.predict()
        if (actual_category == predicted_category):
            self.prev_rule_index = self.current_rule_index  # same rule used
        else:
            # for rule Ri that was active on trial n
            ri_weight = self.saliences[self.current_rule_index] + self.gamma

            # choose rule at random Rj
            rj_index =  choice([i for i in range(0,len(self.rules)) if i not in [self.current_rule_index]])
            rj_weight = self.saliences[rj_index] + self.poisson_dist()

            # set other weights equal to saliences
            self.weights = np.copy(self.saliences)
            self.weights[self.current_rule_index] = ri_weight
            self.weights[rj_index] = rj_weight

            # calculating p_n+1 for rule k
            sum_salience = np.sum(self.saliences)

            next_prob_dist = self.saliences / sum_salience

            self.prev_rule_index = self.current_rule_index

            #select next rule EXCEPT ri and rj
            blackList = [self.current_rule_index, rj_index]
            mask = np.zeros(next_prob_dist.size, dtype = bool)
            mask[blackList] = True
            whiteList = np.ma.array(next_prob_dist, mask=mask)
            next_rule = np.argmax(whiteList)
            self.current_rule_index = next_rule
        self.prev_prediction = predicted_category
        self.prev_trial = trial
        return predicted_category


    def get_trials_to_criterion(self, threshold: int):
        output = np.append(self.trials, self.output, 1)
        seq_trials = 0
        for i, output_row in enumerate(output):
            true_category, _, _, pred_category, _ = output_row
            if true_category == pred_category:
                seq_trials += 1
                if seq_trials == threshold:
                    return i + 1 
            else:
                seq_trials = 0
        return 0