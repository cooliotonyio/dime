import numpy as np
import pickle

fname = 'data/nuswide_metadata/All_Tags.txt'

with open(fname) as f:
    content = f.readlines()

n = len(content)
tag_matrix = [None] * n

for line, idx in zip(content, range(n)):
    tag_matrix[idx] = line.split(' ')[1:]

pickle.dump(tag_matrix, open('pickles/nuswide_metadata/tag_matrix.p', 'wb'))
