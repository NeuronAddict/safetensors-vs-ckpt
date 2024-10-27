import pickle
import fastai

with open('/tmp/danger.pkl', 'rb') as f:
    result = pickle.load(f)
    dir(result)
