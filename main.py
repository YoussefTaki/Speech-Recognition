

from load_data import get_data
from Models import CNN_LSTM
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':
    
    
    data_path=pd.read_csv(r"C:\Users\youss\Desktop\Speech Recognition\Code\RavTess.csv")
    X,Y=get_data(data_path, aug=True)
    
    Emotions = pd.DataFrame(X)
    Emotions['Emotions'] = Y
    
    
    Df= Emotions.dropna(axis=1)
    features=Df.drop('Emotions', axis=1)
    labels=Df["Emotions"]
    # Split the data into training and temporary sets (80% training, 20% temp)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(features,labels, test_size=0.2, random_state=42)

    # Further split the temporary set into training and validation sets (75% training, 25% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)
        
    input_shape = (X_train.shape[1], 1)  # Shape of input features
    num_classes = len(labels.Emotions.unique())     # Number of emotion classes

    # Create CNN model
    model = CNN_LSTM(input_shape, num_classes)

    # Compile the model with Adam optimizer
    adam_optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train_encoded, batch_size=32, epochs=100, validation_data=(X_test, y_test_encoded))

    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)