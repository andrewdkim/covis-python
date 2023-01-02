from Explicit import Explicit
from Procedural import Procedural
import matplotlib.pyplot as plt
import numpy as np
from ExplicitRules import spatial_and_orientation, spatial, orientation



class COVIS:
    def __init__(self, trials, use_explicit=True, use_procedural=True, category_a=1, category_b=2):
        self.trials = trials
        self.delta_oc = 0.01
        self.delta_oe = 0.04
        self.category_a = category_a
        self.category_b = category_b
        self.procedural_model = Procedural()
        self.explicit_model = Explicit([spatial, spatial_and_orientation, orientation])
        self.use_procedural = use_procedural
        self.use_explicit = use_explicit
        self.results = []
        self.model_used = []

        self.procedural_weight = 0.01
        self.explicit_weight = 0.99

    def run_trials(self):
        for i, trial in enumerate(self.trials):
            print(i)
            actual_result = int(trial[0])
            if self.use_procedural and self.use_explicit:
                # competition between procedural and explicit
                explicit_prediction, explicit_confidence = self.explicit_model.run_trial(trial)
                procedural_prediction, procedural_confidence, weight = self.procedural_model.run_trial(trial)

                # make decision
                if self.explicit_weight * explicit_confidence > self.procedural_weight * procedural_confidence:
                    self.model_used.append("explicit")
                    self.results.append([actual_result, explicit_prediction])
                else:
                    self.model_used.append("procedural")
                    self.results.append([actual_result, procedural_prediction])

                print("Explicit weight: " + str(self.explicit_weight))
                print("Explicit confidence: " + str(explicit_confidence))
                print("Procedural weight: " + str(self.procedural_weight))
                print("Procedural confidence: " + str(procedural_confidence))
                print("Explicit product: " + str(self.explicit_weight * explicit_confidence))
                print("Procedural product: " + str(self.procedural_weight * procedural_confidence))
                
                if explicit_prediction == actual_result:
                    # if explicit system gives correct response
                    self.explicit_weight = self.explicit_weight + self.delta_oc * (1 - self.explicit_weight)
                else:
                    self.explicit_weight = self.explicit_weight - self.delta_oe * self.explicit_weight
                self.procedural_weight = 1 - self.explicit_weight

            elif self.use_explicit:
                # use only explicit model
                predicted_result, confidence = self.explicit_model.run_trial(trial)
                self.results.append([actual_result, predicted_result])
            elif self.use_procedural:
                # use only procedural model
                predicted_result, confidence, weight = self.procedural_model.run_trial(trial)
                self.results.append([actual_result, predicted_result])
        print(self.model_used)
    
    """
    Useful function for visualizing how the procedural model is training
    """
    def visualize_procedural_weights(self, weights):
        striatal1_weights = weights[:, 0].reshape((100, 100))
        striatal2_weights = weights[:, 1].reshape((100, 100))
        plt.imshow(striatal1_weights, cmap='viridis')
        plt.colorbar()
        plt.imshow(striatal2_weights, cmap='viridis')
        plt.colorbar()
        plt.show()

    """
    For visualizing how well the COVIS model is trained after running trials
    """
    def visualize_batch_accuracy(self, batch_size=50):
        #graphing learning rate
        results = np.array(self.results)
        num_batch = int(len(results) / batch_size)
        x = []
        y = []
        for i in range(num_batch):
            batch = results[i : i + batch_size]
            actual_categories = batch[:, 0]
            predicted_categories= batch[:, 1]
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
        plt.title("COVIS Learning Curve of Each Batch and its Accuracy")  # add title
        plt.xticks(np.arange(len(x)))
        plt.show()

