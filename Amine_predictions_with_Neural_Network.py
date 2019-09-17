from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import time
start_time = time.time()

# plt.rcParams['figure.figsize'] = [10, 5]

# load the dataset
df = pd.read_csv('amine2.csv')                          # Making the excel file a pandas dataframe
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df = df.drop('AmineU', 1)               # Should always be dropped as we don't want another U value in our inputs

df = df.resample('1T').mean()    # Downsampling the dataset, so that we now have the mean of every 1 minute measurements
df = df.dropna()

# if you wish to slice the data set into specific test splits:
'''
# Slicing the dataset into a test with fouling
df = df.loc['2019-07-08 18:30:00':'2019-07-09 08:00:00']            # ex. Test nr 2
print(df.shape)
'''

# Every column in this data set:
# If you wish to remove some of the columns, it can be done here easily by using # to make it not dropped
df = df.drop('TT229', 1)              # Temperature for Amine - IN
df = df.drop('TIC207', 1)               # Temperature for Amine - Manual - OUT
df = df.drop('FT202Temp', 1)          # Temperature for Amine - OUT
df = df.drop('FT202Flow', 1)          # Flow for Amine
df = df.drop('FT202density', 1)         # Density for Amine
df = df.drop('PDT203', 1)             # Pressure Difference over HX-206 - Amine
df = df.drop('TT404', 1)                # Temperature for cooling medium - OUT
df = df.drop('TIC201', 1)             # Temperature for cooling medium - IN
# df = df.drop('FT400', 1)                #
df = df.drop('TT206', 1)                #
df = df.drop('TIC220', 1)               #
df = df.drop('TIC231', 1)               #
df = df.drop('ProsessdT', 1)            #
df = df.drop('KjolevanndT', 1)          #
df = df.drop('dT1', 1)                  #
df = df.drop('dT2', 1)                  #
df = df.drop('PIC203', 1)               #
df = df.drop('PT213', 1)                #
df = df.drop('HX400PV', 1)              #
df = df.drop('HX400output', 1)          #
print(df.shape)

# Use this code if you want to visualize any parameters:
'''
fig, ax1 = plt.subplots()
ax1.set_title('Pressure Difference vs U-value')
color = 'darkgreen'
ax1.set_xlabel('Date')
ax1.set_ylabel('Pressure Difference [mbar]', color=color)
ax1.plot(df.index, df['PDT203'], color=color, linestyle='none', marker='o')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(1, axis='y')

color = 'darkred'
ax2 = ax1.twinx()
ax2.set_ylabel('U-value [W/m^2*K]', color=color)
ax2.plot(df.index, df['KjolevannU'], color=color, linestyle='none', marker='o')
ax2.tick_params(axis='y', labelcolor=color)
plt.show()
'''

# split into input (X) and output (y) variables
X = np.array(df.drop(['KjolevannU'], 1))                    # Drops "KjolevannU" as feature
y = np.array(df['KjolevannU'])                              # Sets "KjolevannU" as label

# Splitting the data set into training and testing for the algorithms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=None)

# Scaling the data is necessary to avoid errors regarding disproportional values
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
samples = X_train.shape[0]


# The first Neural Network: Multi-Layer Perceptron
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1500)
mlp.fit(X_train, y_train)                                       # Fitting the model

pred_train = mlp.predict(X_train)
pred_test = mlp.predict(X_test)
r2_train = r2_score(y_train, pred_train)                        # Getting scores for how good the model is fitted
r2_test = r2_score(y_test, pred_test)
print('R2 score train:', r2_train)
print('R2 score test:', r2_test)
print('MSE:', mean_squared_error(y_test, pred_test))

# To plot the predictions on the correct timeline, it has to be converted to the same format as the
# rest of the dataframe. Combining predicted training and testing to get the correct amount of data points:
trained = pred_train
predicted = pred_test
total = np.concatenate((trained, predicted), axis=0)    # Combining the predicted training and testing to 1 series
df['pred'] = total                                  # Making the predictions a column in the dataframe


# Plotting was difficult in this case. To get good plots, you need to specify when the training set in the dataframe is
# done, and changing color for the test predictions. This is not a very pythonic way of doing things and I would like
# to change that, but haven't found a sufficient solution.
plt.plot(df['KjolevannU'], color='blue', label='U Value')       # Plotting the "reference values" -  the label.
plt.plot(df['pred'][:samples], 'r-', label='Training')     # samples must be changed from 112261 to plot the reduced dataset
plt.plot(df['pred'][samples:], 'g-', label='Predicted')
plt.xlabel('Time')
plt.ylabel('U-value')
plt.legend(loc=1)
plt.title('Neural Network - MLP')
plt.show()
plt.plot(df['KjolevannU'][samples:], color='blue', label='U Value')  # Making another plot that only contains test parts
plt.plot(df['pred'][samples:], 'g-', label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('U-value')
plt.legend(loc=1)
plt.title('Predicted values for MLP')
plt.show()

df = df.drop('pred', 1)         # Dropping the pred, making a new one in the Sequential model, don't think this is
# necessary as the "pred" column should be overwritten next time it is made


# 2nd Neural Network - Keras Sequential Model
# To play with different settings: Optimizer, no. epochs and no. batch_size can be changed to you liking
model = Sequential()
model.add(Dense(25, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)      # verbose = 1 if you want to inspect every epoch

# evaluate the keras model
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print('R2 score train:', r2_train)
print('R2 score test:', r2_test)

# Same procedure as with the MLP for plotting
trained = pred_train
predicted = pred_test
total = np.concatenate((trained, predicted), axis=0)
df['pred'] = total


plt.plot(df['KjolevannU'], color='blue', label='U Value')
plt.plot(df['pred'][:samples], 'r-', label='Training')
plt.plot(df['pred'][samples:], 'g-', label='Predicted')
plt.xlabel('Time')
plt.ylabel('U-value')
plt.legend(loc=1)
plt.title('FeedForward Neural Network - KSM')
plt.show()
plt.plot(df['KjolevannU'][samples:], color='blue', label='U Value')
plt.plot(df['pred'][samples:], 'g-', label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('U-value')
plt.legend(loc=1)
plt.title('Predicted values for KSM')
plt.show()

print("My program took", time.time() - start_time, "seconds to run")

# This code can also plot the graphs, but not with the time series index
'''
fig = plt.subplots()
plt.plot(np.linspace(1, len(y_train), len(y_train)), y_train, 'b')
plt.plot(np.linspace(1, len(pred_train), len(pred_train)), pred_train, 'g')
plt.title('Training vs. measured')
fig = plt.subplots()
plt.plot(np.linspace(1, len(y_test), len(y_test)), y_test, 'b')
plt.plot(np.linspace(1, len(pred_test), len(pred_test)), pred_test, 'g')
plt.title('Test vs. measured') 
plt.show()
'''