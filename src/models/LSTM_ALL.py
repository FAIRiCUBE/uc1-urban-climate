import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt



def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load your dataset here
    # Assuming you have a CSV file with two columns: 'Year' and 'Population'
    data = pd.read_csv('CompleteData.csv')
    ME = []
    RE = []

    for i in data.index:
        # Extract the 'Population' column as the target variable for prediction
        population_timeSeries = data.iloc[i]
        population_timeSeries = population_timeSeries.drop(['index', 'indic_code', 'urau_code'])
        population_data = population_timeSeries.values.reshape(-1, 1)

        # Normalize the data to a range of (0, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        population_data_scaled = scaler.fit_transform(population_data)

        # Split the data into training and testing sets
        train_size = int(len(population_data_scaled) * 0.7)  # 70% for training, 30% for testing
        train_data, test_data = population_data_scaled[:train_size], population_data_scaled[train_size:]

        # Function to create input sequences and corresponding outputs for LSTM

        # Define the sequence length (number of time steps to look back)
        sequence_length = 1

        # Create sequences for LSTM
        X_train, y_train = create_sequences(train_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        # Reshape the input data for LSTM (samples, time steps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(sequence_length, 1)))  # 50 LSTM units
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer with 1 unit

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Print model summary
        #print(model.summary())

        # Train the model
        epochs = 100
        batch_size = 4

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate the model on the test data
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score}')

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Inverse transform the predictions to get actual population values
        y_pred_actual = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y_test)
        model.save("Models/LSTM_" + data['urau_code'].iloc[i]+"_SLength" + sequence_length.__str__() + "_Epochs" + epochs.__str__() + "_Batch" + batch_size.__str__() +".h5")

        # Calculate and print the Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        rmse = math.sqrt(np.mean(np.abs((y_test_actual - y_pred_actual) ** 2)))

        print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

        print(f'Root Mean Squared Percentage Error (RMSE): {rmse:.2f}')

        ME.append(mape)
        RE.append(rmse)
        #Plot
        x = range(1991, 2022)



        #Accuracy on Training
        y_pred_train = model.predict(X_train)
        y_pred_train_actual = scaler.inverse_transform(y_pred_train)
        y_train_actual = scaler.inverse_transform(y_train)
        plt.figure(figsize=(12, 6))

        plt.plot(x[sequence_length:train_size], y_train_actual, label='Actual (Training)')
        plt.plot(x[sequence_length:train_size], y_pred_train_actual, label='Predicted (Training)')

        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title('Actual and Predicted Population Time Series')
        plt.legend()
        plt.grid(True)
        plt.savefig("Plots/" + data['urau_code'].iloc[i] + "_"+ sequence_length.__str__() + "_Epochs" + epochs.__str__() + "_Batch" + batch_size.__str__() + "_Train.png")




        plt.figure(figsize=(12, 6))

        plt.plot(x[train_size + sequence_length:], y_test_actual, label='Actual (Testing)')
        plt.plot(x[train_size + sequence_length:], y_pred_actual, label='Predicted (Testing)')

        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title('('+ data['urau_code'].iloc[i] + ', RMSE= ' + rmse.__str__()  + ', Sequence_Length= ' + sequence_length.__str__()  + ')')
        plt.legend()
        plt.grid(True)
        plt.savefig("Plots/" + data['urau_code'].iloc[i] + "_"+ sequence_length.__str__() + "_Epochs" + epochs.__str__() + "_Batch" + batch_size.__str__() + "_Test.png")
        #plt.show()



    print(ME)
    print(RE)
    result = np.asarray(RE)
    print("Min: ",result.min())
    print("Max: ", result.max())
    print("Mean: ", result.mean())
    print("STD: ", result.std())

