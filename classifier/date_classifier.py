import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_excel('date_fruit.xlsx')

print(data.shape)
print(data['Class'].unique())

# Create the features dataset
X = data.drop('Class', axis=1)

# Create the labels dataset
y = data.loc[:,'Class']

# Normalize the features dataset and assign it to a variable
X_scaled = minmax_scale(X)

# Create a DataFrame using the new variable
X = pd.DataFrame(X_scaled)
# print(X.head())

# Create an LabelEncoder object
encoder = LabelEncoder()

# Convert string classes to integers using fit_transform() method
y = encoder.fit_transform(y)

# Create X_train, y_train and X_temporary and y_temporary datasets from X and y
X_train, X_temporary, y_train, y_temporary = train_test_split(X, y, train_size=0.8)

# Create validation and test datasets
X_val, X_test, y_val, y_test = train_test_split(X_temporary, y_temporary, train_size=0.5)

# Check the lengths of the X, X_train, X_val and X_test
print(f'Length of the dataset: {len(X)}')
print(f'Length of the training dataset: {len(X_train)}')
print(f'Length of the validation dataset: {len(X_val)}')
print(f'Length of the test dataset: {len(X_test)}')


# Create a model object
model = tf.keras.Sequential()

# Create an input Layer
input_layer = tf.keras.layers.Dense(4096, input_shape=(34,), activation='relu')

# Add input layer to model object
model.add(input_layer)

# Add the first hidden layer with 4096 nodes and relu activation function
model.add(tf.keras.layers.Dense(4096, activation='relu'))
# Add 0.5 dropout
model.add(tf.keras.layers.Dropout(0.5))

# Add 2nd hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# Add 3rd hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# Add 4th hidden layer
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# Add output layer
model.add(tf.keras.layers.Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 100 epochs
results = model.fit(X_train, y_train, epochs=100, validation_data=(X_val,y_val))

# Plot the training and validation loss
plt.plot(results.history['loss'], label='Train')
plt.plot(results.history['val_loss'], label='Test')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()
plt.show()

# Evaluate the performance
test_result = model.test_on_batch(X_test, y_test)
print(test_result)
