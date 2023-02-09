import cv2
import numpy as np
from matplotlib import pyplot as plt

l = cv2.imread('airportL.jpg')
r = cv2.imread('airportR.jpg')
l_new = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
r_new = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
depth = stereo.compute(l_new, r_new)

cv2.imshow('LEFT', l)
cv2.imshow('RIGHT', r)
plt.imshow(depth)
plt.axis('off')
plt.show()
cv2.waitKey(0)