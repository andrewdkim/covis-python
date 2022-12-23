import numpy as np
import matplotlib.pyplot as plt
from random import choice



class Explicit:
    def __init__(self, trials, rules, delta_c = 0.0025, delta_e = 0.02, gamma = 1, lam = 5, category_a = 1, category_b = 2) -> None:
        self.rules = rules
        self.trials = trials
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
        self.n = 1  # current trial
        # initially, all salience is set to 0.25, assuming binary rules
        self.saliences = np.full(len(self.rules), 0.25)
        self.weights = np.zeros(len(self.rules))
        self.output = []
        self.output_saliences = []

    def poisson_dist(self):
        return np.random.poisson(lam=self.lam)

    def predict(self):
        return self.rules[self.current_rule_index](self.trials[self.n - 1])

    # should only be called in n => 2

    def update_salience(self):
        assert self.n >= 2, "Salience updating should only occur on trial 2 an onwards"
        # correct previous category, given by trials
        prev_category = int(self.trials[self.n - 2][0])
        if (self.prev_prediction == prev_category):
            self.saliences[self.prev_rule_index] = self.saliences[self.prev_rule_index] + self.delta_c
        else:
            print(self.delta_e)
            self.saliences[self.prev_rule_index] = self.saliences[self.prev_rule_index] - self.delta_e

    def run_trials(self):
        for i in range(len(self.trials)):
            if self.n >= 2:
                self.update_salience()
            pred_category = self.predict()
         
            self.output.append([pred_category, self.current_rule_index])
            self.output_saliences.append(np.copy(self.saliences))
            if (self.trials[self.n - 1][0] == pred_category):
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
            self.prev_prediction = pred_category
            self.n += 1
    


    def generate_output(self, txt_file_path, salience_file_path, batch_size=50):
        #saving output to txt file
        output = np.append(self.trials, self.output, 1)
        np.savetxt(txt_file_path, output, fmt='%1.3f')
        np.savetxt(salience_file_path, self.output_saliences, fmt='%1.3f')

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