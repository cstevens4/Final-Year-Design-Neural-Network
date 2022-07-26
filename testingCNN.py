import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.layers import Dense
import numpy as np
import cv2

def readImage(img):
    image = cv2.imread(img, 0)
    resize = cv2.resize(image, (28, 28))
    invert = cv2.bitwise_not(resize)

    arr = np.array(invert)
    arr = arr.reshape(1,28, 28, 1)
    arr = arr / 255

    #print("converted")
    return arr

model = load_model("FYD_CNN")

#zero
input = model.predict(readImage("Nathaniel 0.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 0")
print(str(guess) + "\t" + str(conf))

#one
input = model.predict(readImage("Nathaniel 1.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 1")
print(str(guess) + "\t" + str(conf))

#two

#three
input = model.predict(readImage("Nathaniel 3.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 3")
print(str(guess) + "\t" + str(conf))

#three
input = model.predict(readImage("connor 3.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 3")
print(str(guess) + "\t" + str(conf))

#four

#five
input = model.predict(readImage("Nathaniel 5.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 5")
print(str(guess) + "\t" + str(conf))

#five
input = model.predict(readImage("connor 5.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 5")
print(str(guess) + "\t" + str(conf))

#six
input = model.predict(readImage("Nathaniel 6.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 6")
print(str(guess) + "\t" + str(conf))
#seven

#eight
input = model.predict(readImage("Nathaniel 8.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 8")
print(str(guess) + "\t" + str(conf))

#eight
input = model.predict(readImage("connor 8.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 8")
print(str(guess) + "\t" + str(conf))

#eight
input = model.predict(readImage("desmos 8.JPG"))[0]
guess = np.argmax(input)
conf = max(input)
print("Expected: 8")
print(str(guess) + "\t" + str(conf))


#pred_five = model.predict(five)[0]

#print(model.predict(five)[0])
