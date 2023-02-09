import numpy as np
import cv2 as cv
cap = cv.VideoCapture('vtest.avi')
fgbg = cv.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    cv.imshow("feed", frame)
    cv.imshow("bgsubframe", fgmask)
    k = cv.waitKey(0)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

