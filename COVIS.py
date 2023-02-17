from Explicit import Explicit
from Procedural import Procedural
import matplotlib.pyplot as plt
import numpy as np
from ExplicitRules import spatial_and_orientation, spatial, orientation
from progress.bar import ChargingBar


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
        self.procedural_output_weight = None

    def run_trials(self):
        bar = ChargingBar('Running trials', max=len(self.trials), suffix='%(percent)d%%' + " [%(index)d / %(max)d]")
        for i, trial in enumerate(self.trials):
            actual_result = int(trial[0])
            if self.use_procedural and self.use_explicit:
                # competition between procedural and explicit
                explicit_prediction, explicit_confidence = self.explicit_model.run_trial(trial)
                procedural_prediction, procedural_confidence, weight = self.procedural_model.run_trial(trial)

                # print("confidence: Procedural: " + str(procedural_confidence) + ", Explicit: " + str(explicit_confidence))

                # make decision
                if self.explicit_weight * abs(explicit_confidence) > self.procedural_weight * abs(procedural_confidence):
                    self.model_used.append("explicit")
                    # print("explicit - correct" if explicit_prediction == actual_result else "explicit - incorrect")
                    self.results.append([actual_result, explicit_prediction])
                else:
                    self.model_used.append("procedural")
                    # print("procedural - correct" if procedural_prediction == actual_result else "procedural - incorrect")
                    self.results.append([actual_result, procedural_prediction])
                
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
                self.procedural_output_weight = weight
                self.results.append([actual_result, predicted_result])
            bar.next()
        bar.finish()
    
    """
    Useful function for visualizing how the procedural model is training
    """
    def visualize_procedural_weights(self, save_path: str):
        weights = self.procedural_output_weight
        fig, axes = plt.subplots(nrows=1, ncols=2)
        striatal1_weights = weights[:, 0].reshape((100, 100))
        striatal2_weights = weights[:, 1].reshape((100, 100))
        im = axes.flat[0].imshow(striatal1_weights, cmap='viridis', vmin= 0, vmax = 1)
        im = axes.flat[1].imshow(striatal2_weights, cmap='viridis', vmin = 0, vmax= 1)
        axes[0].set_title("Strital A")
        axes[1].set_title("Strital B")
        alpha = self.procedural_model.alpha
        alpha_w = self.procedural_model.alpha_w
        beta_w = self.procedural_model.beta_w
        gamma_w = self.procedural_model.gamma_w
        sigma_p = self.procedural_model.sigma_p
        parameters = "Parameters - trials: " + str(len(self.trials)) + ", alpha: " + str(alpha) + ", alpha_w: " + str(alpha_w) + ", beta_w: " + str(beta_w) + ", gamma_w: " + str(gamma_w) + ", sigma_p: " + str(sigma_p)
        fig.suptitle("Procedural Model Striatal Weights \n(" + parameters + ")")
        fig.set_size_inches(11, 7)
        plt.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(save_path)
        plt.close()


    """
    For visualizing how well the COVIS model is trained after running trials
    """
    def visualize_batch_accuracy(self, save_path: str, batch_size=50):
        #graphing learning rate
        results = np.array(self.results)
        num_batch = int(len(results) / batch_size)
        x = []
        y = []
        for i in range(num_batch):
            batch = results[i * batch_size : i * batch_size + batch_size]
            actual_categories = batch[:, 0]
            predicted_categories= batch[:, 1]
            num_correct = 0
            for j,  predicted_category in enumerate(predicted_categories):
                if predicted_category == actual_categories[j]:
                    num_correct += 1
            x.append(i)
            y.append(num_correct / batch_size)

        plt.plot(x, y)
        plt.xlabel("Batch")  # add X-axis label
        plt.ylabel("Accuracy")  # add Y-axis label

        if self.use_explicit and self.use_procedural:
            plt.title("COVIS Model Accuracy Per Batch (Batch Size = " + str(batch_size) + ")")
        elif self.use_explicit:
            plt.title("Explicit Model Accuracy Per Batch (Batch Size = " + str(batch_size) + ")")
        elif self.use_procedural:
            plt.title("Procedural Model Accuracy Per Batch (Batch Size = " + str(batch_size) + ")")
        plt.xticks(np.arange(len(x)))
        plt.savefig(save_path)
        plt.close()


