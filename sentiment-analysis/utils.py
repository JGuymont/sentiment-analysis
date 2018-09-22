import csv
import pickle
import codecs

def load_csv(path):
    out_data = []
    with open(path, 'r', encoding='cp1252') as csvfile:
        in_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in in_data:
            out_data.append(' '.join(row))
    return out_data

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return