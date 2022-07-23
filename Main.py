import numpy as np
from Explicit import *






def open_data(filepath: str):
  with open(filepath, "r") as file:
    data = []
    for entry in file:
      data.append(np.array(list(map(float, entry.split()))))
    return np.array(data)

def rules():
  def spatial(entry):
    [_, spatial, _] = entry
    return 1 if spatial > -20 else 2

  def spatial_and_orientation(entry):
    [_, spatial, ori] = entry
    return 1 if spatial > -20 and ori > 11.25 else 2

  
  def orientation(entry):
    [_, _, ori] = entry
    return 1 if ori > 11.25 else 2

  return [spatial, spatial_and_orientation, orientation]


def main():
  data = open_data("data/RB_triangle_v1.txt")
  explicit = Explicit(data, rules())
  explicit.run_trials()

if __name__ == "__main__":
  main()