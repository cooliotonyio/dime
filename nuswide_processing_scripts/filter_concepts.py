import pickle
import csv

with open("../nuswide_metadata/Concepts81.txt") as f:
    reader = csv.reader(f)
    concepts = [i[0] for i in list(reader)]

word_vec = pickle.load(open("../pickles/word_embeddings/word_embeddings_tensors.p", "rb"))

base_loader = [[label, word_vec[label]] for label in concepts]

pickle.dump(base_loader, open("../pickles/nuswide_metadata/base_loader.p", "wb"))
