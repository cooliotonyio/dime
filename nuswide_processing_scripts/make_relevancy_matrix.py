import numpy as np
import pickle
import os

path = 'nuswide_metadata/AllLabels/'
n = 269648
relevancy_matrix = np.zeros((n,81), dtype=int)
filenames = []

for idx, filename in enumerate(os.listdir(path)):
    filenames.append(filename)
filenames.sort()

for idx, filename in enumerate(filenames):
    with open(path + filename) as f:
        content = f.readlines()
        curr_column = np.array([int(i[0]) for i in content], dtype=int)
    relevancy_matrix[:, idx] = curr_column

pickle.dump(relevancy_matrix, open('pickles/nuswide_metadata/relevancy_matrix.p', 'wb'))
