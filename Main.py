import numpy as np
from Explicit import *


def open_data(filepath: str):
  with open(filepath, "r") as file:
    data = []
    for entry in file:
      data.append(np.array(list(map(float, entry.split()))))
    return np.array(data)

def main():
  data = open_data("data/RB_triangle_v1.txt")
  # explicit = Explicit()
  


if __name__ == "__main__":
  main()