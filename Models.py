
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation,LSTM


def lstm_cnn(input_shape, num_classes):
# Define the combined model
    model = Sequential()

# Add a 1D CNN layer
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    
# Add a 1D CNN layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
# Add a 1D CNN layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

# Add an LSTM layer
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))

# Flatten the output of the LSTM layer
    model.add(Flatten())

# Fully Connected (FC) layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))  # Assuming 8 emotion classes
    
    return model



def Base_Model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv1D(64, 5, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(256, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


