from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflowjs as tfjs
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

def generateAdditionalData(size, spread):
    sample_sepal_length = np.random.normal(5.00600, 0.35249/spread, size)
    sample_sepal_width = np.random.normal(3.428000, 0.379064/spread, size)
    sample_petal_length = np.random.normal(1.462000, 0.173664/spread, size)
    sample_petal_width = np.random.normal(0.2460003, 0.105386/spread, size)

    hstack1 = np.vstack((sample_sepal_length, 
                        sample_sepal_width,
                        sample_petal_length,
                        sample_petal_width)).T

    sample_sepal_length = np.random.normal(5.936000, 0.516171/spread, size)
    sample_sepal_width = np.random.normal(2.770000, 0.313798/spread, size)
    sample_petal_length = np.random.normal(4.260000, 0.469911/spread, size)
    sample_petal_width = np.random.normal(1.326000, 0.197753/spread, size)

    hstack2 = np.vstack((sample_sepal_length, 
                        sample_sepal_width,
                        sample_petal_length,
                        sample_petal_width)).T

    sample_sepal_length = np.random.normal(6.58800, 0.63588/spread, size)
    sample_sepal_width = np.random.normal(2.974000, 0.322497/spread, size)
    sample_petal_length = np.random.normal(5.552000, 0.551895/spread, size)
    sample_petal_width = np.random.normal(2.02600, 0.27465/spread, size)

    hstack3 = np.vstack((sample_sepal_length, 
                        sample_sepal_width,
                        sample_petal_length,
                        sample_petal_width)).T

    hstack = np.vstack((hstack1, hstack2, hstack3))

    dataset = pd.DataFrame({'sepal.length': hstack[:, 0],
                            'sepal.width': hstack[:, 1],
                            'petal.length': hstack[:, 2], 
                            'petal.width': hstack[:, 3]})

    plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c='red')
    plt.show()

    return dataset