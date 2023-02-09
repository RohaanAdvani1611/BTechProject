import cv2
import numpy as np
cap = cv2.VideoCapture('Firetest.mp4')

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (1000, 600))
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower, upper = [18, 50, 50], [35, 255, 255]
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    firesize = cv2.countNonZero(mask)
    if int(firesize) > 3000:
        cv2.putText(frame, 'Fire Detected', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    if int(firesize) > 8000:
        cv2.putText(frame, 'Explosion Detected', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.imshow('Output', frame)
    cv2.waitKey(1)