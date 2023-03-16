import numpy as np
from COVIS import COVIS
from Explicit import *
from Procedural import *
import pandas as pd
import seaborn
from torch.utils.data import Dataset
from Datasets import ImageDataset, TriangleDataset

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


def train_procedural_model(dataset:Dataset, repeat_count = 1):
    print("Training procedural model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(dataset, use_explicit=False)
        covis_model.run_trials()
        covis_model.visualize_procedural_weights("output/" + str(i + 1) + "_procedural_striatal_weights.png", scale=False)
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_procedural_accuracy.png")

def train_explicit_model(dataset:Dataset, repeat_count = 1):
    print("Training explicit model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(dataset, use_procedural=False)
        covis_model.run_trials()
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_explicit_accuracy.png")

def train_covis_model(dataset:Dataset, repeat_count = 1):
    print("Training COVIS model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(dataset)
        covis_model.run_trials()
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_covis_accuracy.png")
        covis_model.visualize_procedural_usage("output/" + str(i + 1) + "_procedural_usage.png")
        covis_model.export_covis_info("output/" + str(i + 1) + "_covis_info.txt")


def train_procedural_vgg_model(dataset:Dataset, repeat_count = 1):
    print("Training Procedural - VGG model")
    for i in range(repeat_count):
        print("Running: " + str(i + 1) + "/" + str(repeat_count))
        covis_model = COVIS(dataset, use_explicit = False, use_vgg=True)
        covis_model.run_trials()
        covis_model.visualize_batch_accuracy("output/" + str(i + 1) + "_vgg_procedural_accuracy.png")
        covis_model.visualize_procedural_weights("output/" + str(i + 1) + "_vgg_procedural_striatal_weights.png", scale=False)



if __name__ == "__main__":
    # trials = open_data("data/triangle/II_triangle_v1_large.txt")
    # rescaled_trials = rescale_trials(trials)
    # train_procedural_model(rescaled_trials, repeat_count=3)

    IIDataset = ImageDataset("data/II")
    train_procedural_vgg_model(IIDataset)
    # print(IIDataset[0])

    # IITriangleDataset = TriangleDataset("data/triangle/II_triangle_v1.txt", repeat = 20)
    # train_procedural_model(IITriangleDataset)
    