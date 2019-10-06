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
import sys
import importlib
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()

EPOCHS = 100
BATCH_SIZE = 10
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 0

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

MODEL_FOLDER = 'tfjs_models'

def getSequentialModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(10, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(5, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def train(df, column, specification):
    print('Training on the following data:')
    print(df)
    print('-----------------')
    
    EPOCHS, BATCH_SIZE, TEST_SIZE, SHUFFLE, VERBOSE, ACTIVATION, LOSS, OPTIMIZER, METRICS = specification.getParams()
    
    models = []
    columns = 1
    start_time = time.time()
    print(f"Column: {column}")
    X = df.drop(column, axis=1)
    y = df[column]

    r2_train = -100
    r2_test = -100
    run = 1
    while (r2_test < 0.5 and run < 11):
        print(f'Run {run}')
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=TEST_SIZE,
                                                            shuffle=SHUFFLE
                                                            )

        model = specification.getModel(X_train.shape[1])
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
        model.fit(X_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=VERBOSE
                    )

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        print('R2 score train:', r2_train)
        print('R2 score test:', r2_test)
        run += 1

    if r2_test < 0.5:
        print(f'Training for {column} was unsuccessful')
    else:
        tup = (model, column)
        models.append(tup)
        print(f'Training completed in {time.time() - start_time} seconds')

    print("---------------")

    return models

def saveModels(models, folder, subfolder):
    for model, column in models:
        print(f'Saving column {column} in {folder}')
        tfjs_target_dir = "{}\\{}\\{}".format(folder, subfolder, column)
        tfjs.converters.save_keras_model(model, tfjs_target_dir)
        model.save('rig.h5')
    return True

def main(filename, subfolder, column, specification):
    df = pd.read_csv(filename)
    models = train(df, column, specification)
    saved = saveModels(models, MODEL_FOLDER, subfolder)

    if (saved):
        print(f'All models were saved successfully in folder {MODEL_FOLDER}/{subfolder}')
    else:
        print(f'Saving was unsuccessful')
    print(f'Program completed')

# usage: python training_single.py iris_mod.csv testfolder sepal.width specifications.basic
if __name__ == "__main__":
    filename = sys.argv[1]
    subfolder = sys.argv[2]
    column = sys.argv[3]
    specification = importlib.import_module(sys.argv[4])
    main(filename, subfolder, column, specification)

# TODO: rewrite with argparse to allow optional arguments
# https://stackoverflow.com/questions/28479543/run-python-script-with-some-of-the-argument-that-are-optional

# TODO: rewrite _single so that it is a specific case of _multiple