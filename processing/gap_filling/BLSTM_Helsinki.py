import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
import math
import matplotlib.pyplot as plt
from measurer import Measurer
from types import ModuleType
import time
import os
# Function to create input sequences and corresponding outputs for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def BLSTM():
    # Path where the data are stored (the use of the disk in this path is measured).
    # Use '/' to measure the entire disk.
    data_path = '/'
    measurer = Measurer()
    tracker = measurer.start(data_path=data_path)
    # example -> shape = [5490, 2170]
    shape = []

    #Start

    data = pd.read_csv('CompleteData.csv')
    data.head()
    city_index = 8
    population_timeSeries = data.iloc[city_index] # Corresponds to Helsenki

    population_timeSeries = population_timeSeries.drop(['index', 'indic_code', 'urau_code'])
    population_data = population_timeSeries.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    population_data_scaled = scaler.fit_transform(population_data)

    train_size = int(len(population_data_scaled) * 0.7)  # 70% for training, 30% for testing
    train_data, test_data = population_data_scaled[:train_size], population_data_scaled[train_size:]



    # Define the sequence length (number of time steps to look back)
    sequence_length = 3
    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Reshape the input data for LSTM (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(Bidirectional(LSTM(50, input_shape=(sequence_length, 1))))  # 50 LSTM units
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer with 1 unit
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 100
    batch_size = 4
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Inverse transform the predictions to get actual population values
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)

    # Calculate and print the Mean Absolute Percentage Error (MAPE) and RMSE
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    rmse = math.sqrt(np.mean(np.abs((y_test_actual - y_pred_actual) ** 2)))

    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    print(f'Root Mean Squared Percentage Error (RMSE): {rmse:.2f}')
    # Make predictions on the training data
    y_pred_train = model.predict(X_train)
    y_pred_train_actual = scaler.inverse_transform(y_pred_train)
    y_train_actual = scaler.inverse_transform(y_train)

    x = range(1991, 2022)

    # On Training
    plt.figure(figsize=(12, 6))
    plt.plot(x[sequence_length:train_size], y_train_actual, label='Actual (Training)')
    plt.plot(x[sequence_length:train_size], y_pred_train_actual, label='Predicted (Training)')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title('Actual and Predicted Population Time Series')
    plt.legend()
    plt.grid(True)
    plt.savefig('BLSTMTrain.pdf')

    # On Testing
    plt.figure(figsize=(12, 6))
    plt.plot(x[train_size + sequence_length:], y_test_actual, label='Actual (Testing)')
    plt.plot(x[train_size + sequence_length:], y_pred_actual, label='Predicted (Testing)')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title('Actual and Predicted Population Time Series')
    plt.legend()
    plt.grid(True)
    pplt.savefig('BLSTMTest.pdf')

    #End

    # it is very important to use program_path = __file__
    measurer.end(tracker=tracker,
                 shape=shape,
                 libraries=[v.__name__ for k, v in globals().items() if type(v) is ModuleType and not k.startswith('__')],
                 data_path=data_path,
                 program_path=__file__,
                 variables=locals(),
                 csv_file='benchmarks.csv')

if __name__ == "__main__":
    BLSTM()
