import numpy as np


class Explicit:
    def __init__(self, trials, rules) -> None:
        self.rules = rules
        self.trials = trials
        self.delta_c = 0.0025
        self.delta_e = 0.02
        self.gamma = 1
        self.lam = 5
        self.category_a = 1
        self.category_b = 2
        self.ci = 22.5 #just for RB_triangle_v1
        self.epsilon = 0 

        self.prev_rule_index = None
        # when starting out, rule Ri is randomly chosen
        self.current_rule_index = np.random.randint(0, len(rules))
        self.prev_prediction = None
        self.n = 1  # current trial
        # initially, all salience is set to 0.25, assuming binary rules
        self.saliences = np.full(len(self.rules), 0.25)
        self.weights = np.zeros(len(self.rules))
        self.output = []

    def poisson_dist(self):
        return np.random.poisson(lam=self.lam)

    def predict(self):
        return self.rules[self.current_rule_index](self.trials[self.n - 1])
        


    # should only be called in n => 2
    def update_salience(self):
        assert self.n >= 2, "Salience updating should only occur on trial 2 an onwards"
        # correct previous category, given by trials
        prev_category = self.trials[self.n - 2][0]
        if (self.prev_prediction == prev_category):
            self.saliences[self.current_rule_index] = self.saliences[self.prev_rule_index] + self.delta_c
        else:
            self.saliences[self.current_rule_index] = self.saliences[self.prev_rule_index] - self.delta_e

    def run_trials(self):
        for i in range(len(self.trials)):
            if self.n >= 2:
                self.update_salience()
            pred_category = self.predict()
            self.output.append([pred_category, self.current_rule_index])
            if (self.trials[self.n - 1][0] == pred_category):
                self.prev_rule_index = self.current_rule_index  # same rule used
            else:
                # for rule Ri that was active on trial n
                ri_weight = self.saliences[self.current_rule_index] + self.gamma

                # choose rule at random Rj
                rj_index = np.random.randint(0, len(self.rules))
                rj_weight = self.saliences[rj_index] + self.poisson_dist()

                # set other weights equal to saliences
                self.weights = self.saliences
                self.weights[self.current_rule_index] = ri_weight
                self.weights[rj_index] = rj_weight

                # calculating p_n+1 for rule k
                sum_salience = np.sum(self.saliences)
                next_prob_dist = self.saliences / sum_salience #TODO: the update rule is not 100% accurate

                # not sure about this part, but I choose the max of all probabilities
                self.prev_rule_index = self.current_rule_index
                next_rule = np.argmax(next_prob_dist)
                self.current_rule_index = next_rule
            self.prev_prediction = pred_category
            self.n = self.n + 1
    
    def generate_output():
        pass
