import numpy as np
def ideal_weights():
    num_sensory = 10000
    striatal_1 = []
    for i in range(int(num_sensory / 2)):
        striatal_1.append([1])
    for i in range(int(num_sensory / 2), num_sensory):
        striatal_1.append([0])
    striatal_2 = []
    for i in range(int(num_sensory / 2)):
        striatal_2.append([0])
    for i in range(int(num_sensory / 2), num_sensory):
        striatal_2.append([1])
    return np.hstack((np.array(striatal_1), np.array(striatal_2)))
