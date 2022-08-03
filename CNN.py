import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import cv2

#data prep
data = mnist.load_data()

(x_train, y_train), (x_test, y_test) = data

x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#CNN
model = keras.Sequential(
    [
        keras.Input(shape=(28,28,1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

#model.summary()

#Training
batch_size = 128
epochs = 10

#model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#print("Model is complete")

#model.save("FYD_CNN")
#Results
#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])

print(x_train.shape[-3:])
print(x_test.shape)