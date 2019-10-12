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
import sys
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

standardScaler_x = joblib.load('std_sc_x.bin')

def main(modelname, testfile, target):
    df = pd.read_csv(testfile)
    X = df.drop(target, axis=1)
    y = df[target]

    X_std = standardScaler_x.transform(X)

    model = load_model(modelname)
    pred = model.predict(X_std)

    time = range(df.shape[0])

    print(df)

    plt.plot(time, y, color='red')
    plt.plot(time, pred, color='blue')
    plt.show()

# usage: python testing.py rig_standardized.h5 ../datasets/rig_feil.csv Load
if __name__ == "__main__":
    modelname = sys.argv[1]
    testfile = sys.argv[2]
    target = sys.argv[3]
    main(modelname, testfile, target)