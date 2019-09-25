from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import tensorflowjs as tfjs
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
start_time = time.time()

tf.compat.v1.disable_eager_execution()

# load the dataset
df = pd.read_csv('iris.csv') 
df = df.drop('variety', 1)
columns = len(df.iloc[0,:])

X = df.iloc[:, 0:columns-1]
Y = df.iloc[:, columns-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=None)

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
samples = X_train.shape[0]

model = load_model('my_model.h5')

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print('R2 score train:', r2_train)
print('R2 score test:', r2_test)