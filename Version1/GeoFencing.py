import cv2
import imutils
import HandTrackingModule as htm
import numpy as np

def BuildFence(img, shape, color, pos, dim, p, cent, rad):
    if shape == 'rectangle':
        cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + dim[0], pos[1] + dim[1]), color, 3)
    if shape == 'triangle':
        p1, p2, p3 = p[:]
        cv2.line(img, p1, p2, color, 3)
        cv2.line(img, p2, p3, color, 3)
        cv2.line(img, p1, p3, color, 3)
    if shape == 'circle':
        cv2.circle(img, cent, rad, color, 3)
    return img

def EditFence(frame, shape, edit_var, pos, dim, p, cent, rad, a):
    if shape == 'rectangle':
        if edit_var == 'X':
            if a == '+':
                pos[0] -= 1
            if a == '-':
                pos[0] += 1
        if edit_var == 'Y':
            if a == '+':
                pos[1] += 1
            if a == '-':
                pos[1] -= 1
        if edit_var == 'W':
            if a == '+':
                dim[0] -= 1
            if a == '-':
                dim[0] += 1
        if edit_var == 'H':
            if a == '+':
                dim[1] -= 1
            if a == '-':
                dim[1] += 1
    if shape == 'triangle':
        p1, p2, p3 = p[:]
        p1 = list(p1)
        p2 = list(p2)
        p3 = list(p3)
        if edit_var == 'X1':
            if a == '+':
                p1[0] -= 1
            if a == '-':
                p1[0] += 1
        if edit_var == 'Y1':
            if a == '+':
                p1[1] += 1
            if a == '-':
                p1[1] -= 1
        if edit_var == 'X2':
            if a == '+':
                p2[0] -= 1
            if a == '-':
                p2[0] += 1
        if edit_var == 'Y2':
            if a == '+':
                p2[1] += 1
            if a == '-':
                p2[1] -= 1
        if edit_var == 'X3':
            if a == '+':
                p3[0] -= 1
            if a == '-':
                p3[0] += 1
        if edit_var == 'Y3':
            if a == '+':
                p3[1] += 1
            if a == '-':
                p3[1] -= 1
        p1 = tuple(p1)
        p2 = tuple(p2)
        p3 = tuple(p3)
        p = [p1, p2, p3]
    if shape == 'circle':
        cent = list(cent)
        if edit_var == 'CX':
            if a == '+':
                cent[0] -= 1
            if a == '-':
                cent[0] += 1
        if edit_var == 'CY':
            if a == '+':
                cent[1] += 1
            if a == '-':
                cent[1] -= 1
        if edit_var == 'R':
            if a == '+':
                rad -= 1
            if a == '-':
                rad += 1
        cent = tuple(cent)
    return pos, dim, p, cent, rad

# img = cv2.imread('airport.jpg')
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.75)

# Default Coordinates: Fence Dimensions
shape = 'rectangle'
edit_var = 'X'
# Rectangle: x, y, w, h
pos = [100, 100]
dim = [500, 200]
# Triangle: p1, p2, p3
p = [(750, 400), (500, 500), (300, 300)]
# Circle: cent, rad
cent = (400, 400)
rad = 200
color = (255, 0, 0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Controller Design
    cv2.rectangle(frame, (0, 0), (1280, 70), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (10, 10), (420, 60), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'RECTANGLE', (15, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.rectangle(frame, (200, 15), (250, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'X', (215, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (255, 15), (305, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'Y', (270, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (310, 15), (360, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'W', (325, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (365, 15), (415, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'H', (380, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)

    cv2.rectangle(frame, (430, 10), (850, 60), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'TRIANGLE', (435, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.rectangle(frame, (585, 15), (625, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'X1', (587, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)
    cv2.rectangle(frame, (630, 15), (670, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'Y1', (632, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)
    cv2.rectangle(frame, (675, 15), (715, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'X2', (677, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)
    cv2.rectangle(frame, (720, 15), (760, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'Y2', (722, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)
    cv2.rectangle(frame, (765, 15), (805, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'X3', (767, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)
    cv2.rectangle(frame, (810, 15), (848, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'Y3', (812, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 2)

    cv2.rectangle(frame, (860, 10), (1270, 60), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'CIRCLE', (865, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.rectangle(frame, (980, 15), (1040, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'CX', (987, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (1050, 15), (1110, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'CY', (1057, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (1120, 15), (1265, 55), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, 'RADIUS', (1132, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)

    cv2.rectangle(frame, (430, 60), (850, 80), (255, 255, 255), cv2.FILLED)

    cv2.rectangle(frame, (0, 100), (50, 150), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (5, 105), (45, 145), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, '-', (13, 137), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)
    cv2.rectangle(frame, (1230, 100), (1280, 150), (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (1235, 105), (1275, 145), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, '+', (1243, 137), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 128), 3)

    frame = BuildFence(frame, shape, color, pos, dim, p, cent, rad)

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=True)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        fingers = detector.fingersUp()
        if fingers[1] == 1:
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            if 10 < y1 < 60:
                if 10 < x1 < 420:
                    cv2.rectangle(frame, (10, 10), (420, 60), (255, 0, 255), 3)
                    shape = 'rectangle'
                    if 15 < y1 < 55:
                        if 200 < x1 < 250:
                            cv2.rectangle(frame, (200, 15), (250, 55), (128, 128, 0), 2)
                            edit_var = 'X'
                        if 255 < x1 < 305:
                            cv2.rectangle(frame, (255, 15), (305, 55), (128, 128, 0), 2)
                            edit_var = 'Y'
                        if 310 < x1 < 360:
                            cv2.rectangle(frame, (310, 15), (360, 55), (128, 128, 0), 2)
                            edit_var = 'W'
                        if 365 < x1 < 415:
                            cv2.rectangle(frame, (365, 15), (415, 55), (128, 128, 0), 2)
                            edit_var = 'H'
                if 430 < x1 < 850:
                    cv2.rectangle(frame, (430, 10), (850, 60), (255, 0, 255), 3)
                    shape = 'triangle'
                    if 15 < y1 < 55:
                        if 585 < x1 < 625:
                            cv2.rectangle(frame, (585, 15), (625, 55), (128, 128, 0), 2)
                            edit_var = 'X1'
                        if 630 < x1 < 670:
                            cv2.rectangle(frame, (630, 15), (670, 55), (128, 128, 0), 2)
                            edit_var = 'Y1'
                        if 675 < x1 < 715:
                            cv2.rectangle(frame, (675, 15), (715, 55), (128, 128, 0), 2)
                            edit_var = 'X2'
                        if 720 < x1 < 760:
                            cv2.rectangle(frame, (720, 15), (760, 55), (128, 128, 0), 2)
                            edit_var = 'Y2'
                        if 765 < x1 < 805:
                            cv2.rectangle(frame, (765, 15), (805, 55), (128, 128, 0), 2)
                            edit_var = 'X3'
                        if 810 < x1 < 848:
                            cv2.rectangle(frame, (810, 15), (848, 55), (128, 128, 0), 2)
                            edit_var = 'Y3'
                if 860 < x1 < 1270:
                    cv2.rectangle(frame, (860, 10), (1270, 60), (255, 0, 255), 3)
                    shape = 'circle'
                    if 15 < y1 < 55:
                        if 980 < x1 < 1040:
                            cv2.rectangle(frame, (980, 15), (1040, 55), (128, 128, 0), 2)
                            edit_var = 'CX'
                        if 1050 < x1 < 1110:
                            cv2.rectangle(frame, (1050, 15), (1110, 55), (128, 128, 0), 2)
                            edit_var = 'CY'
                        if 1120 < x1 < 1265:
                            cv2.rectangle(frame, (1120, 15), (1265, 55), (128, 128, 0), 2)
                            edit_var = 'R'
            if 105 < y1 < 145:
                if 5 < x1 < 45:
                    cv2.rectangle(frame, (5, 105), (45, 145), (128, 128, 0), 2)
                    pos, dim, p, cent, rad = EditFence(frame, shape, edit_var, pos, dim, p, cent, rad, '+')
                if 1235 < x1 < 1275:
                    cv2.rectangle(frame, (1235, 105), (1275, 145), (128, 128, 0), 2)
                    pos, dim, p, cent, rad = EditFence(frame, shape, edit_var, pos, dim, p, cent, rad, '-')

    cv2.putText(frame, shape + ' : ' + edit_var, (560, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    cv2.imshow('IMAGE', frame)
    cv2.waitKey(1)

