import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./csvdump/output.csv')

# Define neutral baselines for pitch and MFCCs (these values are examples and should be validated or adjusted)
neutral_baselines = {
    'pitch_mean': 2000,
    'mfcc_0': -250,
    'mfcc_1': 120,
    'mfcc_2': -30,
    'mfcc_3': 40,
    'mfcc_4': 0,
    'mfcc_5': 1,
    'mfcc_6': -1,
    'mfcc_7': -20,
    'mfcc_8': 0.5,
    'mfcc_9': -12,
    'mfcc_10': -6,
    'mfcc_11': -3,
    'mfcc_12': -11
}

# Calculate the absolute deviation from neutral baselines
for feature, baseline in neutral_baselines.items():
    data[f'adjustment_{feature}'] = np.abs(data[feature] - baseline)

# Create a single 'adjustment_needed' column by averaging all adjustment values
adjustment_columns = [f'adjustment_{feature}' for feature in neutral_baselines.keys()]
data['adjustment_needed'] = data[adjustment_columns].mean(axis=1)

# Drop intermediate adjustment columns as they are no longer needed after averaging
data.drop(columns=adjustment_columns, inplace=True)

# Prepare the feature matrix X and target vector y
X = data.drop(['filename', 'frame', 'adjustment_needed'], axis=1)
y = data['adjustment_needed']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and compile the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression, no activation function
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test_scaled, y_test)

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model for later use
model.save('accent_modification_model.keras')
