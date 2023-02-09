import cv2
import numpy as np
#capture video in wecam and store frames in cap variable
cap=cv2.VideoCapture(0)

#read frames from cap variable store first frame in frame1 and frame2 respectively
ret, frame1 = cap.read()
ret, frame2 = cap.read()
#start loop in order to show all frames of video
while cap.isOpened():
    #cv2.absdiff() is a method used to compare two images/frames
    diff = cv2.absdiff(frame1, frame2)
    #convert the difference calculated to grayscale in order to apply threshold on it
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #Gaussian blur a widely used effect in graphics software, typically to reduce image noise and reduce detail.
    # Applying a Gaussian blur has the effect of reducing the image's high-frequency components
    # it allows us to provide different weight kernel in x and y directions
    #here kernel is set as (5,5) which is the value pixels are compared to
    blur= cv2.GaussianBlur(gray, (5,5), 0)
    #apply binary threshold on grayscale difference
    # here all pixels of intensity less than 20 will become black and those with more than 20 will become white
    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    #dilate threshold image to fill in all the holes
    #With binary images, dilation connects areas that are separated by spaces smaller than the structuring element
    #iterations is the number of times we want to perform dilation of our image
    dil= cv2.dilate(th, None, iterations=3)
    #find the contours in this thresholded image
    #RETR_TREE is the contour mode which is a hierarchy type ,such that contours present inside other contours have higher heirarchy
    #CHAIN_APPROX_NONE is a contour approximation method
    contours, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #draw rectangles around any found contour
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #display video
    cv2.imshow("feed", frame1)
    #update frame to next frame
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(40)==27:
        break

cap.release()
cv2.destroyAllWindows()
