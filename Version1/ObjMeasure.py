import cv2
import numpy as np
import utils
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    _, img = cap.read()
    imgContours, finalContours = utils.getcontours(img, minArea=20000, filter=4)
    if len(finalContours) != 0:
        for obj in finalContours:
            cv2.polylines(imgContours, [obj[2]], True, (0,255,0), 2)
            nPoints = utils.reorder(obj[2])
            nW = round((utils.findDis(nPoints[0][0], nPoints[1][0])/10),1)
            nH = round((utils.findDis(nPoints[0][0], nPoints[2][0])/ 10), 1)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]), (255,0,255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgContours, '{}cm'.format(nW), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,255), 2)
            cv2.putText(imgContours, '{}cm'.format(nH), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)
        cv2.imshow('IC2', imgContours)
    cv2.waitKey(1)