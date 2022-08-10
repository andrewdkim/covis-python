import numpy as np
from Explicit import *
from ExplicitRules import spatial_and_orientation, spatial, orientation






def open_data(filepath: str):
  with open(filepath, "r") as file:
    data = []
    for entry in file:
      data.append(np.array(list(map(float, entry.split()))))
    return np.array(data)




def main():
  data = open_data("data/RB_triangle_v1_large.txt")
  explicit = Explicit(data, [spatial, spatial_and_orientation, orientation])
  explicit.run_trials()
  explicit.generate_output("output/explicit_output.txt", "output/saliences.txt")
  

if __name__ == "__main__":
  main()