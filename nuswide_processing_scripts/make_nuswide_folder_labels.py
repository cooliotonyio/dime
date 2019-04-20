import csv
import pickle

with open('./data/labels.csv') as f:
    reader = csv.reader(f)
    NUS_WIDE_classes = [i[0] for i in list(reader)]
for i in range(len(NUS_WIDE_classes)):
    if '_' in NUS_WIDE_classes[i]:
        NUS_WIDE_classes[i] = NUS_WIDE_classes[i].split('_')[0]
    if NUS_WIDE_classes[i] == 'adobehouses':
        NUS_WIDE_classes[i] = 'adobe'
    if NUS_WIDE_classes[i] == 'kauai':
        NUS_WIDE_classes[i] = 'hawaii'
    if NUS_WIDE_classes[i] == 'oahu':
        NUS_WIDE_classes[i] = 'hawaii'
        
pickle.dump(NUS_WIDE_classes, open("./pickles/nuswide_metadata/folder_labels.p", "wb"))