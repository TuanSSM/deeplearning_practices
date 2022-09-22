import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the cifar-10 dataset in Keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Print the number of samples 
print(f'X_train: {len(X_train)}')
print(f'X_test: {len(X_test)}')
print(f'y_train: {len(y_train)}')
print(f'y_test: {len(y_test)}')

plt.imshow(X_test[789])

print(X_test[789].shape)

# Create the validation datasets 
# and assign the last 10000 images of X_train and y_train
X_val = X_train[40000:]
y_val = y_train[40000:]

# Create new train datasets
# and assign the first 40000 images of X_train and y_train
X_train = X_train[:40000]
y_train = y_train[:40000]

# Print the lengths of the each dataset
print(f'X_train: {len(X_train)}')
print(f'X_val: {len(X_val)}')
print(f'X_test: {len(X_test)}')

# Divide each dataset by 255
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

# Create a model object
model = tf.keras.Sequential()

# Add a convolution and max pooling layer
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding='same',
                                 activation='relu',
                                 input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

# Add more convolution and max pooling layers
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(3,3),
                                 strides=(1,1),
                                 padding='same',
                                 activation='relu'))

# Flatten the convolution layer
model.add(tf.keras.layers.Flatten())

# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

# Add the output layer
model.add(tf.keras.layers.Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 50 epochs with batch size of 128
results = model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(X_val, y_val))

# Plot the the training and validation loss
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.show()

# Plot the the training and validation accuracy
plt.plot(results.history["accuracy"], label="accuracy")
plt.plot(results.history["val_accuracy"], label="val_accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# Evaluate the performance
model.evaluate(X_test, y_test)

# Make prediction on the reshaped sample
prediction_result = model.predict(X_test[789].reshape(1,32,32,3))

# Print the prediction result
prediction_result

# Find the predicted class and probability
predicted_class = prediction_result.argmax()
predicted_probability = prediction_result.max()

print(f'This image belongs to class {predicted_class} with {predicted_probability} probability %')
