import numpy as np
import pickle

relevancy_matrix = pickle.load(open('pickles/nuswide_metadata/relevancy_matrix.p', 'rb'))
fname = "data/nuswide_metadata/Concepts81.txt"

with open(fname) as f:
    idx_to_concept = f.readlines()

for idx, line in enumerate(idx_to_concept):
    idx_to_concept[idx] = line.split('\n')[0]

n = relevancy_matrix.shape[0]
concept_matrix = [None] * n
neg_concept_matrix = [None] * n

for idx, line in enumerate(relevancy_matrix):
    concepts = []
    neg_concepts = []
    for count, indicator in enumerate(line):
        if indicator == 1:
            concepts.append(idx_to_concept[count])
        else:
            neg_concepts.append(idx_to_concept[count])
    concept_matrix[idx] = concepts
    neg_concept_matrix[idx] = neg_concepts

pickle.dump(concept_matrix, open('pickles/nuswide_metadata/concept_matrix.p', 'wb'))
pickle.dump(neg_concept_matrix, open('pickles/nuswide_metadata/neg_concept_matrix.p', 'wb'))
