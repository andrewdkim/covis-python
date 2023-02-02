import numpy as np
from COVIS import COVIS
from Explicit import *
from Procedural import *
import pandas as pd
import seaborn

def open_data(filepath: str):
    with open(filepath, "r") as file:
        data = []
        for entry in file:
            data.append(np.array(list(map(float, entry.split()))))
        return np.array(data)


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

def graph_trial_data(trials):
    """
    For visualizing triangle_v1 data
    """
    categories = map(lambda num: "A" if num == 1 else "B", trials[:, 0])
    spa = trials[:, 1]
    ori = trials[:, 2]
    df = pd.DataFrame(dict(Spatial=spa, Orientation=ori, Categories = categories))
    seaborn.scatterplot(x='Spatial', y='Orientation', data=df, hue='Categories')
    plt.show()


def train_procedural_model(trials, repeat_count = 1):
    print("Training procedural model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(trials, use_explicit=False)
        covis_model.run_trials()
        covis_model.visualize_procedural_weights("output/" + str(i + 1) + "_procedural_striatal_weights.png")
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_procedural_accuracy.png")

def train_explicit_model(trials, repeat_count = 1):
    print("Training explicit model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(trials, use_procedural=False)
        covis_model.run_trials()
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_explicit_accuracy.png")

def train_covis_model(trials, repeat_count = 1):
    print("Training COVIS model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(trials)
        covis_model.run_trials()
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_covis_accuracy.png")

if __name__ == "__main__":
    trials = open_data("data/II_triangle_v1_large.txt")
    rescaled_trials = rescale_trials_RB_triangle(trials)
    train_procedural_model(rescaled_trials, repeat_count=3)
