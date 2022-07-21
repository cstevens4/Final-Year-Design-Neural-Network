import cv2
import numpy as np

img = cv2.imread("connor 5.JPG", 0)

new_img = cv2.resize(img, (28,28))

invert = cv2.bitwise_not(new_img)

vec = np.ravel(invert)

print(img.shape)
print(new_img.shape)

#print(new_img)
#print(invert)

#cv2.imshow("inverted", invert)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(vec.size)