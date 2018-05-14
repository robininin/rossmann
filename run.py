import sys
sys.path.append("/anaconda3/envs/udacity_tensorflow_py3/lib/python3.5/site-packages")
import pickle
import numpy as np
import math
import pandas as pd
from model_adj_embedding import NN_with_embedding
from keras.models import load_model

num_network = 1
testing = True

#X_train is 0.97 training set. X_all_train is the whole training set
with open("X_train.pickle","rb") as f:
    (X_train, target_train) = pickle.load(f)

with open("X_all_train.pickle","rb") as f:
    (X_all_train, target_all) = pickle.load(f)

if testing:
    X, y = X_all_train, target_all
else:
    X, y = X_train, target_train

hist = []
for i in range(num_network):
    model = NN_with_embedding(X, y, testing)
    print("model {} is fitting...".format(i+1))
    model.fit()
    model.model.save("/Users/Robin/Desktop/udacity_ml/Rossman_store/robin/models/model_18/model{}.h5".format(i))
    hist.append(model.hist.history)

with open("/Users/Robin/Desktop/udacity_ml/Rossman_store/robin/models/model_18/history1.pickle","wb") as f:
    pickle.dump(hist,f,-1)

