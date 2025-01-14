import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print("The file was not found. Please check the filepath.")
        return None

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])  # Create a sequence of data
        y.append(data[i + sequence_length])      # Append the next value to predict
    return np.array(X), np.array(y)

def prepare_data(filepath, sequence_length):
    
    data = load_data(filepath)
    if data is not None:
        if 'Price' in data.columns:
            data = data['Price'].values  
            X, y = create_sequences(data, sequence_length)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            return X_train, y_train, X_test, y_test
        else:
            print("Column 'Price' not found in the data. Please check the column names.")
            return None, None, None, None
    else:
        return None, None, None, None

# Data preparation
filepath = 'data_preprocessed.csv'
sequence_length = 10
X_train, y_train, X_test, y_test = prepare_data(filepath, sequence_length)

# Debugging: Ensure data is prepared correctly
if X_train is not None and y_train is not None:
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Ensure the shape is appropriate for LSTM input
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and compile the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
else:
    print("Failed to prepare training data.")
