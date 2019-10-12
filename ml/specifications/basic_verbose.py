from keras.models import Sequential
from keras.layers import Dense

EPOCHS = 100
BATCH_SIZE = 16
TEST_SIZE = 0.2
SHUFFLE = True
VERBOSE = 1

ACTIVATION = 'relu'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'
METRICS = ['mean_squared_error']

def getModel(INPUT_DIM):
    model = Sequential()
    model.add(Dense(10, input_dim=INPUT_DIM, activation=ACTIVATION))
    model.add(Dense(5, activation=ACTIVATION))
    model.add(Dense(1, activation=ACTIVATION))
    return model

def getParams():
    return (
        EPOCHS,
        BATCH_SIZE,
        TEST_SIZE,
        SHUFFLE,
        VERBOSE,
        ACTIVATION,
        LOSS,
        OPTIMIZER,
        METRICS
    )