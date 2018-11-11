import pickle

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
