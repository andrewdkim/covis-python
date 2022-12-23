import numpy as np
from COVIS import COVIS
from Explicit import *
from Procedural import *
from ExplicitRules import spatial_and_orientation, spatial, orientation


def open_data(filepath: str):
    with open(filepath, "r") as file:
        data = []
        for entry in file:
            data.append(np.array(list(map(float, entry.split()))))
        return np.array(data)

# if criterion is not met, then it will return None


def generate_criterion_graph(control_trial, dual_trial):

    # graphing learning rate
    data = {'control': control_trial, 'dual': dual_trial}
    keys = list(data.keys())
    values = list(data.values())

    plt.bar(keys, values)
    plt.xlabel("Rule-based Control vs Dual ")  # add X-axis label
    plt.ylabel("Trials to Criterion")  # add Y-axis label
    plt.title("COVIS")  # add title
    plt.show()


def rescale_trials_RB_triangle(trials):
    orientation = trials[:, 1]  # "x"
    spatial = trials[:, 2]  # "y"
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
    return np.column_stack((trials[:, 0], orientation, spatial))



def main():
    trials = open_data("data/RB_triangle_v1_large.txt")
    rescaled_trials = rescale_trials_RB_triangle(trials)
    covis_model = COVIS(rescaled_trials, use_explicit = False, use_procedural = True)
    covis_model.run_trials()
    covis_model.visualize_batch_accuracy()

#   criterion_threshold = 8
#   explicit_control= Explicit(data, [spatial, spatial_and_orientation, orientation])
#   explicit_control.run_trials()
#   explicit_control.generate_output("output/explicit_control_output.txt", "output/explicit_control_saliences.txt")
#   control_trial_to_criterion = explicit_control.get_trials_to_criterion(criterion_threshold)

#   explicit_dual= Explicit(data, gamma=20, lam=0.5, rules=[spatial, spatial_and_orientation, orientation])
#   explicit_dual.run_trials()
#   explicit_dual.generate_output("output/explicit_dual_output.txt", "output/explicit_dual_saliences.txt")
#   dual_trial_to_criterion = explicit_dual.get_trials_to_criterion(criterion_threshold)

#   print(control_trial_to_criterion, dual_trial_to_criterion)
#   generate_criterion_graph(control_trial_to_criterion, dual_trial_to_criterion)


if __name__ == "__main__":
    main()
