import pyims
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
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
plt.rcParams['figure.figsize'] = [10, 5]
pd.set_option('display.max_columns', 15)

print(pyims.list_aspen_servers())               # Prints the list of available servers to connect to
print(pyims.list_pi_servers())

c = pyims.IMSClient('GFA', 'Aspen')             # Choosing which server to connect to via pyIMS
c.connect()                                     # Connecting directly to the Aspen server

# Tags provided to Master thesis

tags = ['GFA.27-TT___181_.PV', 'GFA.27-TT___215_.PV', 'GFA.27-PT___180_.PV', 'GFA.40-TT___069_.PV',
        'GFA.40-PT___074_.PV', 'GFA.27-TIC__215_.OUT', 'GFA.27-XV___167_.CMD', 'GFA.27-XV___167_.ZSH',
        'GFA.27-ZT___167_.PV', 'GFA.27-FI___165B.PV']


# All possible tags
'''
tags = ['GFA.27-PT___180_.PV', 'GFA.27-TT___215_.PV', 'GFA.27-TIC__215_.OUT',
        'GFA.40-PDIT_128_.PV', 'GFA.40-TT___049_.PV', 'GFA.40-TIC__046_.PV', 'GFA.27-PDIT_072_.PV',
        'GFA.27-PT___140_.PV', 'GFA.23-TT___646_.PV', 'GFA.40-PT___074_.PV', 'GFA.40-TT___069_.PV',
        'GFA.27-TIT__139_.PV', 'GFA.27-PT___131_.PV', 'GFA.27-TIC__170_.PV', 'GFA.27-PT___171_.PV',
        'GFA.40-FT___057B.PV', 'GFA.27-FI___125B.PV', 'GFA.24-FI___046C.PV', 'GFA.27-XV___167_.CMD',
        'GFA.27-XV___167_.ZSH', 'GFA.27-ZT___167_.PV', 'GFA.27-TT___116_.PV',
        'GFA.23-TIT__607_.PV', 'GFA.27-TIT__206_.PV', 'GFA.27-PDT__126_.PV', 'GFA.27-TT___181_.PV',
        'GFA.27-PIT__048_.PV', 'GFA.27-TT___076_.PV', 'GFA.24-PDIT_010_.PV', 'GFA.23-PIT__644_.PV',
        'GFA.27-PDIT_026_.PV', 'GFA.27-PDI__128_.PV', 'GFA.27-FI___165B.PV']
'''

df = c.read_tags(tags, '01-May-08 13:00:00', '01-Jun-19 13:00:00', 3600)  # 3600 = Seconds between every measurement

# Making a column with known cleaning dates of the HX
# Uses that further to make a date counter that counts days since last cleaning
cleaning_dates = ['01-May-08 15:00:00', '22-Sep-08 00:00:00', '26-Aug-09 00:00:00', '25-Dec-10 00:00:00',
                  '05-Apr-16 00:00:00', '18-Apr-16 00:00:00', '19-Jun-17 00:00:00', '15-Sep-18 00:00:00',
                  '01-Jun-19 13:00:00']

df['cleaning_time'] = cleaning_dates[0]

for i in range(len(cleaning_dates)-1):
    df['cleaning_time'].loc[cleaning_dates[i]: cleaning_dates[i+1]] = cleaning_dates[i]


df['cleaning_time'] = pd.to_datetime(df['cleaning_time'])
df['cleaning_time_til_utc'] = pd.to_datetime(df['cleaning_time'], utc=True)
df['index_til_utc'] = pd.to_datetime(df.index, utc=True)
df['datodifferanse'] = df['index_til_utc'] - df['cleaning_time_til_utc']
df['float_datoteller'] = df['datodifferanse'].dt.total_seconds() / 86400


df = df.drop('cleaning_time_til_utc', 1)                      # 1 = axis
df = df.drop('index_til_utc', 1)
df = df.drop('datodifferanse', 1)
df = df.drop('cleaning_time', 1)

df['U-verdier'] = np.nan                    # Make a new column for input of U-values
# Input specific U-values in cells:
df.at['2008-05-01 13:00:00', 'U-verdier'] = 1885.7
df.at['2010-11-22 21:00:00', 'U-verdier'] = 930.8
df.at['2010-11-28 10:00:00', 'U-verdier'] = 929.3
df.at['2011-03-08 12:00:00', 'U-verdier'] = 1885.7
df.at['2012-08-16 18:00:00', 'U-verdier'] = 1757.1
df.at['2013-06-01 00:00:00', 'U-verdier'] = 1700.3
df.at['2014-09-25 18:00:00', 'U-verdier'] = 1232.6
df.at['2016-05-06 09:00:00', 'U-verdier'] = 1422.7
df.at['2016-05-06 10:00:00', 'U-verdier'] = 1648.3
df.at['2016-11-03 18:00:00', 'U-verdier'] = 1438.6
df.at['2017-06-19 18:00:00', 'U-verdier'] = 988.7
df.at['2017-06-20 15:00:00', 'U-verdier'] = 1140.3
df.at['2017-06-21 09:00:00', 'U-verdier'] = 1163.5
df.at['2017-10-17 14:00:00', 'U-verdier'] = 1208.2
df.at['2018-08-20 16:00:00', 'U-verdier'] = 1128.3
df.at['2018-09-24 11:00:00', 'U-verdier'] = 1439.3
df.at['2019-06-01 13:00:00', 'U-verdier'] = 1200

# Interpolate to get missing U-values in the column:
df['U-interpolated'] = df['U-verdier'].interpolate(method='linear')
df = df.drop('U-verdier', 1)


def clean_dataset():
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = clean_dataset()
# GFA.27-TIC__215_.OUT = Cooling Medium Valve Opening
X = np.array(df.drop(columns=['U-interpolated']))            # Drops "GFA.27-TIC__215_.OUT" as feature
y = np.array(df['U-interpolated'])                           # Sets "GFA.27-TIC__215_.OUT" as label

# Dividing the dataset for training and testing. Shuffle = False, since it's time dependent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=None)
samples = X_train.shape[0]

# Scaling the data is necessary
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)


mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1500)
mlp.fit(X_train, y_train)

pred_train = mlp.predict(X_train)
pred_test = mlp.predict(X_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print('R2 score train:', r2_train)
print('R2 score test:', r2_test)
print('MSE:', mean_squared_error(y_test, pred_test))


trained = pred_train
predicted = pred_test
total = np.concatenate((trained, predicted), axis=0)
df['pred'] = total

plt.plot(df['pred'][:samples], 'r', markevery=500, label='Training')
plt.plot(df['pred'][samples:], 'g', markevery=500, label='Predicted')
plt.plot(df['U-interpolated'], color='blue', label='U-value')
plt.axvline(x='2017-02-21 23:00:00+01:00', ymin=0.0, ymax=1.0)
plt.ylabel('U-value [W/m^2*K]')
plt.xlabel('Time')
plt.title('Neural Network - MLP Regression')
plt.legend(loc=1)
plt.show()
plt.plot(df['U-interpolated'][samples:], color='blue', label='U-value')
plt.plot(df['pred'][samples:], 'g-', label='Predicted')
plt.xlabel('Time')
plt.ylabel('U-value [W/m^2*K]')
plt.legend(loc=1)
plt.title('Predicted values for MLP Regression')
plt.show()

df = df.drop('pred', 1)


# define the keras model
# Feedforward neural-network
model = Sequential()
model.add(Dense(25, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)


# evaluate the keras model
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)
print('R2 score train:', r2_train)
print('R2 score test:', r2_test)

trained = pred_train
predicted = pred_test
total = np.concatenate((trained, predicted), axis=0)
df['pred'] = total


plt.plot(df['U-interpolated'], color='blue', label='U-value')
plt.plot(df['pred'][:samples], 'r-', label='Training')
plt.plot(df['pred'][samples:], 'g-', label='Predicted')
plt.xlabel('Time')
plt.ylabel('U-value [W/m^2*K]')
plt.legend(loc=1)
plt.title('FeedForward Neural Network')
plt.show()
plt.plot(df['U-interpolated'][samples:], color='blue', label='U-value')
plt.plot(df['pred'][samples:], 'g-', label='Predicted')
plt.xlabel('Time')
plt.ylabel('U-value [W/m^2*K]')
plt.legend(loc=1)
plt.title('Predicted values for Feed forward NN')
plt.show()


