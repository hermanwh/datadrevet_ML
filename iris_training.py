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
import generate_iris_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
start_time = time.time()

tf.compat.v1.disable_eager_execution()

df_iris = pd.read_csv('iris.csv')
df_iris = df_iris.drop('variety', 1)
columns = len(df_iris.iloc[0,:])

generatedData = generate_iris_data.generateAdditionalData(100, 2)
df = pd.concat([df_iris, generatedData])
print(df)

EPOCHS = 100
BATCH_SIZE = 10
ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 0

for column in df.columns:
    print(f"Column: {column}")
    X = df.drop(column, axis=1)
    y = df[column]

    r2_train = -100
    r2_test = -100

    run = 1
    while (r2_test < 0.5 and run < 11):
        print(f'Run: {run}')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=SHUFFLE)

        model = Sequential()
        model.add(Dense(10, input_dim=X_train.shape[1], activation=ACTIVATION))
        model.add(Dense(5, activation=ACTIVATION))
        model.add(Dense(1, activation=ACTIVATION))

        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        print('R2 score train:', r2_train)
        print('R2 score test:', r2_test)
        run += 1

    tfjs_target_dir = "tfjs_models\\{}".format(column)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)


    # Code just for IRIS dataset
    if column == "sepal.length" or  column == "sepal.width":
        plt.scatter(X_train.iloc[:, 0], y_train.iloc[:], c='blue')
        plt.scatter(X_test.iloc[:, 0], y_test, c='orange')
        plt.scatter(X_test.iloc[:, 0], pred_test, c='red')
        plt.show()
    elif column == "petal.length" or column == "petal.width":
        plt.scatter(X_train.iloc[:, -1], y_train.iloc[:], c='blue')
        plt.scatter(X_test.iloc[:, -1], y_test, c='orange')
        plt.scatter(X_test.iloc[:, -1], pred_test, c='red')
        plt.show()

    print("---------------")