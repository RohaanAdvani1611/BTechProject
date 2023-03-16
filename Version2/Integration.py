import cv2
import imutils
import HandTrackingModule as htm
import numpy as np
import time
import PoseEstimatorModule as pm
import random
import string
import datetime
import google_streetview.api


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

def random_string(N):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def Two_Dim_Image_Split(img):
    h, w, channels = img.shape
    half = w // 2
    half2 = h // 2
    tl = img[:half2, :half]
    bl = img[half2:, :half]
    tr = img[:half2, half:]
    br = img[half2:, half:]
    parts = [tl, bl, tr, br]
    return parts

def Quadsplit(img, lvl):
    tree, tmp = [[img]], [img]
    for i in range(lvl):
        tmp2 = []
        for j in range(len(tmp)):
            parts = Two_Dim_Image_Split(tmp[j])
            for p in parts:
                tmp2.append(p)
        tmp = tmp2
        tree.append(tmp)
    return tree

def check_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1 = np.array([(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)])
    box2 = np.array([(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)])

    intersection = np.logical_and(box1, box2)

    if np.sum(intersection) > 0:
        return 1
    else:
        return 0
#colours
blue = (255,0,0)
red = (0,0,255)
green = (0,255,0)
orange = (0,128,255)
pink = (255,153,255)
silver = (192,192,192)
purple = (255,0,255)
white = (255,255,255)
black = (0,0,0)
aqua = (255,255,0)
maroon = (0,0,128)
yellow = (0,255,255)
olive = (128, 128, 0)

font = cv2.FONT_HERSHEY_PLAIN
counter = 0
#icons for mainpage
size = 120
icon1 = cv2.imread("QS.png")
icon1 = cv2.resize(icon1, (size, size))
icon1gray = cv2.cvtColor(icon1, cv2.COLOR_BGR2GRAY)
ret, icon1mask = cv2.threshold(icon1gray, 1, 255, cv2.THRESH_BINARY)

icon2 = cv2.imread("download.png")
icon2 = cv2.resize(icon2, (size, size))
icon2gray = cv2.cvtColor(icon2, cv2.COLOR_BGR2GRAY)
ret, icon2mask = cv2.threshold(icon2, 1, 255, cv2.THRESH_BINARY)

icon3 = cv2.imread("custom1.png")
icon3 = cv2.resize(icon3, (size, size))
icon3gray = cv2.cvtColor(icon3, cv2.COLOR_BGR2GRAY)
ret, icon3mask = cv2.threshold(icon3gray, 1, 255, cv2.THRESH_BINARY_INV)

icon4 = cv2.imread("motion.png")
icon4 = cv2.resize(icon4, (size, size))
icon4gray = cv2.cvtColor(icon4, cv2.COLOR_BGR2GRAY)
ret, icon4mask = cv2.threshold(icon4gray, 1, 255, cv2.THRESH_BINARY)

icon5 = cv2.imread("humanbody.png")
icon5 = cv2.resize(icon5, (size, size))
icon5gray = cv2.cvtColor(icon5, cv2.COLOR_BGR2GRAY)
ret, icon5mask = cv2.threshold(icon5gray, 1, 255, cv2.THRESH_BINARY)

icon6 = cv2.imread("Vehicle Detection.png")
icon6 = cv2.resize(icon6, (size, size))
icon6gray = cv2.cvtColor(icon6, cv2.COLOR_BGR2GRAY)
ret, icon6mask = cv2.threshold(icon6gray, 1, 255, cv2.THRESH_BINARY)

icon7 = cv2.imread("Coco.png")
icon7 = cv2.resize(icon7, (size, size))
icon7gray = cv2.cvtColor(icon7, cv2.COLOR_BGR2GRAY)
ret, icon7mask = cv2.threshold(icon7gray, 1, 255, cv2.THRESH_BINARY)

icon8 = cv2.imread("satellite.png")
icon8 = cv2.resize(icon8, (size, size))
icon8gray = cv2.cvtColor(icon8, cv2.COLOR_BGR2GRAY)
ret, icon8mask = cv2.threshold(icon8gray, 1, 255, cv2.THRESH_BINARY)

icon9 = cv2.imread("gun.png")
icon9 = cv2.resize(icon9, (size, size))
icon9gray = cv2.cvtColor(icon9, cv2.COLOR_BGR2GRAY)
ret, icon9mask = cv2.threshold(icon9gray, 1, 255, cv2.THRESH_BINARY)

icon10 = cv2.imread("anomaly.jpg")
icon10 = cv2.resize(icon10, (size, size))
icon10gray = cv2.cvtColor(icon10, cv2.COLOR_BGR2GRAY)
ret, icon10mask = cv2.threshold(icon10gray, 1, 255, cv2.THRESH_BINARY)

icon11 = cv2.imread("violence.png")
icon11 = cv2.resize(icon11, (size, size))
icon11gray = cv2.cvtColor(icon11, cv2.COLOR_BGR2GRAY)
ret, icon11mask = cv2.threshold(icon11gray, 1, 255, cv2.THRESH_BINARY)

icon12 = cv2.imread("Fire Detection.png")
icon12 = cv2.resize(icon12, (size, size))
icon12gray = cv2.cvtColor(icon12, cv2.COLOR_BGR2GRAY)
ret, icon12mask = cv2.threshold(icon12gray, 1, 255, cv2.THRESH_BINARY)

lvl = 3
master_quad = []
tree = []

#customfence global variables
fence_coords = []
custom_x = 0
custom_y = 0
custom_i = 0
cur_point = ()

mode = 'Mainpage'
vid = [0, 'Falltest.mp4', 'Firetest.mp4', 'ak47.mp4', 'vtest.avi','cctv2.mp4','Running.mp4']

if mode == 'Fall':
    cap = cv2.VideoCapture(vid[1])
elif mode == 'Fire':
    cap = cv2.VideoCapture(vid[2])
elif mode == 'Weapon':
    cap = cv2.VideoCapture(vid[3])
elif mode == 'Pedestrian':
    cap = cv2.VideoCapture(vid[4])
elif mode == 'Group':
    cap = cv2.VideoCapture(vid[4])
elif mode == 'Tracker':
    cap = cv2.VideoCapture(vid[5])
elif mode == 'Coco':
    cap = cv2.VideoCapture(vid[5])
else:
    cap = cv2.VideoCapture(vid[0])

if mode == 'Streetview':
    params = [{
        'size': '600x300',
        'location': '46.414382,10.013988',
        'heading': '151.78',
        'pitch': '-0.76',
        'key': '21102001'
    }]

    results = google_streetview.api.results(params)
    results.download_links('C:/Users/dhake/OneDrive/Desktop/BTechProject-main/BTechProject-main/Version1/')


cap.set(3, 1280)
cap.set(4, 720)
handdetector = htm.handDetector(detectionCon=0.75)
posedetector = pm.poseDetector()

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

thres = 0.6
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.readlines()
# print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


fgbg = cv2.createBackgroundSubtractorMOG2()


net2 = cv2.dnn.readNet("C:/Users/dhake/Downloads/Weapon-Detection-with-yolov3-master/Weapon-Detection-with-yolov3-master/weapon_detection/yolov3-master/yolov3-master/yolov3_training_2000.weights", "C:/Users/dhake/Downloads/Weapon-Detection-with-yolov3-master/Weapon-Detection-with-yolov3-master/weapon_detection/yolov3-master/yolov3-master/yolov3_testing.cfg")
classes = ["Weapon"]
layer_names = net2.getLayerNames()
#output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#master_pos = []
master_areas = []
master_speed = []
overcrowdlim = 5

car_cascade = cv2.CascadeClassifier('cars.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

TrDict = {'csrt':cv2.legacy.TrackerCSRT_create}

trackers = cv2.legacy.MultiTracker_create()
ret, frame = cap.read()
k = 3
if mode == "Tracker":
    for i in range(k):
        cv2.imshow('IMAGE', frame)
        bbi = cv2.selectROI('IMAGE', frame)
        tracker_i = TrDict['csrt']()
        trackers.add(tracker_i, frame, bbi)

if mode != 'Streetview':
    while True:
        if mode != 'Motion':
            success, frame = cap.read()
            if mode != 'Tracker':
                frame = cv2.flip(frame, 1)
        frame = handdetector.findHands(frame)
        lmList = handdetector.findPosition(frame, draw=True)
        currenttime = datetime.datetime.now()
        cv2.putText(frame,str(currenttime), (825, 700), font, 2, (0, 255, 0), 2)

        if mode == 'Mainpage':
            cv2.rectangle(frame, (0, 0), (1280, 70), white, cv2.FILLED)
            cv2.putText(frame, 'ANOMALY AND OBJECT DETECTION FOR SECUIRTY SURVEILLANCE', (130, 65), font, 2, (0, 0, 255), 3)
            cv2.rectangle(frame, (60, 120), (200, 260), black, cv2.FILLED)
            cv2.putText(frame, 'QUADSPLIT', (90, 280), font, 1, red, 2)
            icon1roi = frame[130:250, 70:190]
            icon1roi[np.where(icon1mask)] = 0
            icon1roi += icon1            
            cv2.rectangle(frame, (270, 120), (410, 260), black, cv2.FILLED)
            cv2.putText(frame, 'GEOFENCE', (300, 280), font, 1, red, 2)
            icon2roi = frame[130:250, 280:400]
            icon2roi[np.where(icon2mask)] = 0
            icon2roi += icon2
            cv2.rectangle(frame, (480, 120), (620, 260), black, cv2.FILLED)
            cv2.putText(frame, 'TRACKER', (510, 280), font, 1, red, 2)
            icon3roi = frame[130:250, 490:610]
            icon3roi[np.where(icon3mask)] = 0
            icon3roi += icon3
            cv2.rectangle(frame, (690, 120), (830, 260), black, cv2.FILLED)
            cv2.putText(frame, 'MOTION', (730, 280), font, 1, red, 2)
            icon4roi = frame[130:250, 700:820]
            icon4roi[np.where(icon4mask)] = 0
            icon4roi += icon4
            cv2.rectangle(frame, (900, 120), (1040, 260), black, cv2.FILLED)
            cv2.putText(frame, 'PEDESTRIAN', (925, 280), font, 1, red, 2)
            icon5roi = frame[130:250, 910:1030]
            icon5roi[np.where(icon5mask)] = 0
            icon5roi += icon5
            cv2.rectangle(frame, (1110, 120), (1250, 260), black, cv2.FILLED)
            cv2.putText(frame, 'VEHICLE', (1150, 280), font, 1, red, 2)
            icon6roi = frame[130:250, 1120:1240]
            icon6roi[np.where(icon6mask)] = 0
            icon6roi += icon6

            #second row
            cv2.rectangle(frame, (60, 320), (200, 460), black, cv2.FILLED)
            cv2.putText(frame, 'COCO', (100, 480), font, 1, red, 2)
            icon7roi = frame[330:450, 70:190]
            icon7roi[np.where(icon7mask)] = 0
            icon7roi += icon7
            cv2.rectangle(frame, (270, 320), (410, 460), black, cv2.FILLED)
            cv2.putText(frame, 'SATELLITE', (300, 480), font, 1, red, 2)
            icon8roi = frame[330:450, 280:400]
            icon8roi[np.where(icon8mask)] = 0
            icon8roi += icon8
            cv2.rectangle(frame, (480, 320), (620, 460), black, cv2.FILLED)
            cv2.putText(frame, 'WEAPON', (510, 480), font, 1, red, 2)
            icon9roi = frame[330:450, 490:610]
            icon9roi[np.where(icon9mask)] = 0
            icon9roi += icon9
            cv2.rectangle(frame, (690, 320), (830, 460), black, cv2.FILLED)
            cv2.putText(frame, 'ANOMALY', (725, 480), font, 1, red, 2)
            icon10roi = frame[330:450, 700:820]
            icon10roi[np.where(icon10mask)] = 0
            icon10roi += icon10
            cv2.rectangle(frame, (900, 320), (1040, 460), black, cv2.FILLED)
            cv2.putText(frame, 'VIOLENCE', (930, 480), font, 1, red, 2)
            icon11roi = frame[330:450, 910:1030]
            icon11roi[np.where(icon11mask)] = 0
            icon11roi += icon11
            cv2.rectangle(frame, (1110, 320), (1250, 460), black, cv2.FILLED)
            cv2.putText(frame, 'FIRE', (1155, 480), font, 1, red, 2)
            icon12roi = frame[330:450, 1120:1240]
            icon12roi[np.where(icon12mask)] = 0
            icon12roi += icon12

            #handtracking for icons on mainpage
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1:
                    cv2.circle(frame, (x1, y1), 15, red, cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 15, green, cv2.FILLED)
                    if 120 < y1 < 260 and 120 < y2 < 260:
                        if 60 < x1 < 200 and 60 < x2 < 200:
                            cv2.rectangle(frame, (60, 120), (200, 260), purple, 3)
                            mode = 'Quadsplit'
                        if 270 < x1 < 410 and 270 < x2 < 410:
                            cv2.rectangle(frame, (270, 120), (410, 260), purple, 3)
                            mode = 'setfence'
                        if 480 < x1 < 620 and 480 < x2 < 620:
                            cv2.rectangle(frame, (480, 120), (620, 260), purple, 3)
                            #for i in range(k):
                                #cv2.imshow('IMAGE', frame)
                                #bbi = cv2.selectROI('IMAGE', frame)
                                #tracker_i = TrDict['csrt']()
                                #trackers.add(tracker_i, frame, bbi)
                            mode = 'settracker'
                        if 690 < x1 < 830 and 690 < x2 < 830:
                            cv2.rectangle(frame, (690, 120), (830, 260), purple, 3)
                            ret, frame = cap.read()
                            ret, frame2 = cap.read()
                            mode = 'Motion'
                        if 900 < x1 < 1040 and 900 < x2 < 1040:
                            cv2.rectangle(frame, (900, 120), (1040, 260), purple, 3)
                            mode = 'Pedestrian'
                        if 1110 < x1 < 1250 and 1110 < x2 < 1250:
                            cv2.rectangle(frame, (1110, 120), (1250, 260), purple, 3)
                            mode = 'Vehicle'
                                                        
                    if 320 < y1 < 460 and 320 < y2 < 460:
                        if 60 < x1 < 200 and 60 < x2 < 200:
                            cv2.rectangle(frame, (60, 320), (200, 460), purple, 3)
                            mode = 'Coco'
                        if 270 < x1 < 410 and 270 < x2 < 410:
                            cv2.rectangle(frame, (270, 320), (410, 460), purple, 3)
                            mode = 'Satellite'
                        if 480 < x1 < 620 and 480 < x2 < 620:
                            cv2.rectangle(frame, (480, 320), (620, 460), purple, 3)
                            mode = 'Weapon'
                        if 690 < x1 < 830 and 690 < x2 < 830:
                            cv2.rectangle(frame, (690, 320), (830, 460), purple, 3)
                            mode = 'Anomaly'
                        if 900 < x1 < 1040 and 900 < x2 < 1040:
                            cv2.rectangle(frame, (900, 320), (1040, 460), purple, 3)
                            mode = 'Violence'
                        if 1110 < x1 < 1250 and 1110 < x2 < 1250:
                            cv2.rectangle(frame, (1110, 320), (1250, 460), purple, 3)
                            mode = 'Fire'

        if mode == 'Geofence':
            # Controller Design
            cv2.rectangle(frame, (0, 0), (1280, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (10, 10), (420, 60), black, cv2.FILLED)
            cv2.putText(frame, 'RECTANGLE', (15, 45), font, 2, red, 3)
            cv2.rectangle(frame, (200, 15), (250, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'X', (215, 45), font, 2, maroon, 3)
            cv2.rectangle(frame, (255, 15), (305, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'Y', (270, 45), font, 2, maroon, 3)
            cv2.rectangle(frame, (310, 15), (360, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'W', (325, 45), font, 2, maroon, 3)
            cv2.rectangle(frame, (365, 15), (415, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'H', (380, 45), font, 2, maroon, 3)

            cv2.rectangle(frame, (430, 10), (850, 60), black, cv2.FILLED)
            cv2.putText(frame, 'TRIANGLE', (435, 45), font, 2, red, 3)
            cv2.rectangle(frame, (585, 15), (625, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'X1', (587, 45), font, 2, maroon, 2)
            cv2.rectangle(frame, (630, 15), (670, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'Y1', (632, 45), font, 2, maroon, 2)
            cv2.rectangle(frame, (675, 15), (715, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'X2', (677, 45), font, 2, maroon, 2)
            cv2.rectangle(frame, (720, 15), (760, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'Y2', (722, 45), font, 2, maroon, 2)
            cv2.rectangle(frame, (765, 15), (805, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'X3', (767, 45), font, 2, maroon, 2)
            cv2.rectangle(frame, (810, 15), (848, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'Y3', (812, 45), font, 2, maroon, 2)

            cv2.rectangle(frame, (860, 10), (1270, 60), black, cv2.FILLED)
            cv2.putText(frame, 'CIRCLE', (865, 45), font, 2, red, 3)
            cv2.rectangle(frame, (980, 15), (1040, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'CX', (987, 45), font, 2, maroon, 3)
            cv2.rectangle(frame, (1050, 15), (1110, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'CY', (1057, 45), font, 2, maroon, 3)
            cv2.rectangle(frame, (1120, 15), (1265, 55), yellow, cv2.FILLED)
            cv2.putText(frame, 'RADIUS', (1132, 45), font, 2, maroon, 3)

            cv2.rectangle(frame, (430, 60), (850, 80), white, cv2.FILLED)

            cv2.rectangle(frame, (0, 100), (70, 170), white, cv2.FILLED)
            cv2.rectangle(frame, (5, 105), (65, 165), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (18, 150), font, 3, maroon, 3)
            cv2.rectangle(frame, (1210, 100), (1280, 170), white, cv2.FILLED)
            cv2.rectangle(frame, (1215, 105), (1275, 165), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (1227, 150), font, 3, maroon, 3)

            frame = BuildFence(frame, shape, blue, pos, dim, p, cent, rad)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "setfence"
                if fingers[1] == 1:
                    cv2.circle(frame, (x1, y1), 15, red, cv2.FILLED)
                    if 10 < y1 < 60:
                        if 10 < x1 < 420:
                            cv2.rectangle(frame, (10, 10), (420, 60), purple, 3)
                            shape = 'rectangle'
                            if 15 < y1 < 55:
                                if 200 < x1 < 250:
                                    cv2.rectangle(frame, (200, 15), (250, 55), olive, 2)
                                    edit_var = 'X'
                                if 255 < x1 < 305:
                                    cv2.rectangle(frame, (255, 15), (305, 55), olive, 2)
                                    edit_var = 'Y'
                                if 310 < x1 < 360:
                                    cv2.rectangle(frame, (310, 15), (360, 55), olive, 2)
                                    edit_var = 'W'
                                if 365 < x1 < 415:
                                    cv2.rectangle(frame, (365, 15), (415, 55), olive, 2)
                                    edit_var = 'H'
                        if 430 < x1 < 850:
                            cv2.rectangle(frame, (430, 10), (850, 60), purple, 3)
                            shape = 'triangle'
                            if 15 < y1 < 55:
                                if 585 < x1 < 625:
                                    cv2.rectangle(frame, (585, 15), (625, 55), olive, 2)
                                    edit_var = 'X1'
                                if 630 < x1 < 670:
                                    cv2.rectangle(frame, (630, 15), (670, 55), olive, 2)
                                    edit_var = 'Y1'
                                if 675 < x1 < 715:
                                    cv2.rectangle(frame, (675, 15), (715, 55), olive, 2)
                                    edit_var = 'X2'
                                if 720 < x1 < 760:
                                    cv2.rectangle(frame, (720, 15), (760, 55), olive, 2)
                                    edit_var = 'Y2'
                                if 765 < x1 < 805:
                                    cv2.rectangle(frame, (765, 15), (805, 55), olive, 2)
                                    edit_var = 'X3'
                                if 810 < x1 < 848:
                                    cv2.rectangle(frame, (810, 15), (848, 55), olive, 2)
                                    edit_var = 'Y3'
                        if 860 < x1 < 1270:
                            cv2.rectangle(frame, (860, 10), (1270, 60), purple, 3)
                            shape = 'circle'
                            if 15 < y1 < 55:
                                if 980 < x1 < 1040:
                                    cv2.rectangle(frame, (980, 15), (1040, 55), olive, 2)
                                    edit_var = 'CX'
                                if 1050 < x1 < 1110:
                                    cv2.rectangle(frame, (1050, 15), (1110, 55), olive, 2)
                                    edit_var = 'CY'
                                if 1120 < x1 < 1265:
                                    cv2.rectangle(frame, (1120, 15), (1265, 55), olive, 2)
                                    edit_var = 'R'
                    if 105 < y1 < 165:
                        if 5 < x1 < 65:
                            cv2.rectangle(frame, (5, 105), (65, 165), olive, 2)
                            pos, dim, p, cent, rad = EditFence(frame, shape, edit_var, pos, dim, p, cent, rad, '+')
                        if 1215 < x1 < 1275:
                            cv2.rectangle(frame, (1215, 105), (1275, 165), olive, 2)
                            pos, dim, p, cent, rad = EditFence(frame, shape, edit_var, pos, dim, p, cent, rad, '-')
            cv2.putText(frame, shape + ' : ' + edit_var, (560, 75), font, 1, red, 2)
        
        if mode == 'Coco':
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, box, green, thickness=2)
                    a = classNames[classId-1].upper()
                    cv2.putText(frame, a, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, green, 2)
                    cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), font, 1, green, 2)
            
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == 'bg':
            frame = fgbg.apply(frame)
            cv2.rectangle(frame, (0, 0), (200, 70), white, cv2.FILLED)
            cv2.putText(frame, 'MOTION DETECTION', (10, 45), font, 1, red, 2)
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1:
                    if 10 < y1 < 60 and 10 < y2 < 60 :
                        if 0 < x1 < 200 and 0 < x2 < 200:
                            cv2.rectangle(frame, (0, 0), (200, 70), purple, 3)
                            ret, frame = cap.read()
                            ret, frame2 = cap.read()
                            mode = 'Motion'
        
        if mode == 'Motion':
            cv2.rectangle(frame, (0, 0), (300, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (10, 10), (290, 60), black, cv2.FILLED)
            cv2.putText(frame, 'BACKGROUND SUBTRACTION', (20, 45), font, 1, red, 2)
            diff = cv2.absdiff(frame, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur= cv2.GaussianBlur(gray, (5,5), 0)
            _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dil= cv2.dilate(th, None, iterations=3)
            contours, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 700:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
            cv2.imshow("IMAGE", frame)
            frame = frame2
            ret, frame2 = cap.read()
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                x2, y2 = lmList[12][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"
                if fingers[1] == 1 and fingers[2] == 1:
                    if 10 < y1 < 60 and 10 < y2 < 60 :
                        if 10 < x1 < 290 and 10 < x2 < 290:
                            cv2.rectangle(frame, (0, 0), (300, 70), purple, 3)
                            mode = 'bg'


        if mode == 'Fall':
            frame = cv2.resize(frame, (1000, 600))
            frame = posedetector.findPose(frame)
            lmList = posedetector.findPosition(frame)
            depth = posedetector.getDepth()
            #print(depth)
            if len(lmList) != 0:
                if lmList[0][2] < lmList[24][2] and lmList[24][2] < lmList[28][2] and lmList[0][2] < lmList[25][2] and lmList[24][2] < lmList[27][2]:
                    cv2.putText(frame, 'Standing', (10, 10), font, 1, red, 2)
                    begin = time.time()
                if lmList[0][2] > lmList[28][2] or lmList[0][2] > lmList[27][2]:
                    cv2.putText(frame, 'Lying Down', (10, 10), font, 1, red, 2)
                    end = time.time()
                    if (end - begin < 2):
                        cv2.putText(frame, 'Fell Down', (10, 30), font, 1, red, 2)

        if mode == 'Fire':
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
                cv2.putText(frame, 'Fire Detected', (10, 35), font, 2, red, 2)
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == 'Weapon':
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), black, True, crop=False)

            net2.setInput(blob)
            outs = net2.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0: print("weapon detected in frame")
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

        if mode == 'Pedestrian':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
            count = len(bodies)
            if count > overcrowdlim:
                cv2.putText(frame, 'Overcrowding detected', (30, 35), font, 2, red, 2)
            cv2.putText(frame, str(count), (10,35), font, 2, blue, 2)
            for (x,y,w,h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), yellow, 2)
                pos_x = str(int((x + w)/2))
                pos_y = str(int((y + h)/2))
                a = pos_x+" , "+pos_y
                cv2.putText(frame, a, (x, y - 20), font, 1, red, 1)
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == 'Group':
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            areas = []
            for x in boxes:
                areas.append(int(x[2])*int(x[3]))
            smallest = min(areas)  
            for i in range(len(areas)):
                areas[i] -= smallest
            total = sum(areas)
            master_areas.append(total) 
            last1 = master_areas[len(master_areas) - 2]
            last = master_areas[len(master_areas) - 1]
            diff = last - last1
            if diff > 100000:
                cv2.putText(frame, 'Group Meet Detected', (10, 35), font, 2, red, 2)
            elif diff < -100000:
                cv2.putText(frame, 'Group Split Detected', (10, 35), font, 2, red, 2)

            #print(master_areas)
            for (xA, yA, xB, yB) in boxes:
                #remove rectangles with single pedestrian
                cv2.rectangle(frame, (xA, yA), (xB, yB),green, 2)

        if mode == "Quadsplit":
            cv2.rectangle(frame, (0, 0), (200, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (10, 10), (60, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (20, 55), font, 3, red, 3)

            cv2.rectangle(frame, (80, 10), (130, 60), yellow, cv2.FILLED)
            cv2.putText(frame, str(lvl), (90, 55), font, 2, red, 3)

            cv2.rectangle(frame, (140, 10), (190, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (150, 55), font, 3, red, 3)

            cv2.rectangle(frame, (1080, 0), (1280, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (1090, 10), (1270, 60), yellow, cv2.FILLED)
            cv2.putText(frame, 'START', (1100, 55), font, 3, red, 3)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1:
                    cv2.circle(frame, (x1, y1), 15, red, cv2.FILLED)
                    if 10 < y1 < 60:
                        if 10 < x1 < 60:
                            cv2.rectangle(frame, (10, 10), (60, 60), purple, 3)
                            lvl -= 1
                            if lvl < 0:
                                lvl = 0
                        if 140 < x1 < 190:
                            cv2.rectangle(frame, (140, 10), (190, 60), purple, 3)
                            lvl += 1
                            if lvl > 5:
                                lvl = 5
                        if 1090 < x1 < 1270:
                            cv2.rectangle(frame, (1090, 10), (1270, 60), purple, 3)
                            tree = Quadsplit(frame, lvl)
                    
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"
            leafnodes = tree[len(tree)-1]
            for i in leafnodes:
                tmp = []
                #detect objects and anomalies in leaf nodes
                master_quad.append(tmp)

        if mode == "setfence":
            cv2.rectangle(frame, (135, 295), (545, 425), white, cv2.FILLED)
            cv2.rectangle(frame, (140, 300), (540, 420), yellow, cv2.FILLED)
            cv2.putText(frame, 'GEOFENCE', (190, 380), font, 3, red, 3)

            cv2.rectangle(frame, (735, 295), (1145, 425), white, cv2.FILLED)
            cv2.rectangle(frame, (740, 300), (1140, 420), yellow, cv2.FILLED)
            cv2.putText(frame, 'CUSTOM FENCE', (755, 380), font, 3, red, 3)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1:
                    cv2.circle(frame, (x1, y1), 15, red, cv2.FILLED)
                    if 300 < y1 < 420:
                        if 140 < x1 < 540:
                            cv2.rectangle(frame, (140, 300), (540, 420), purple, 3)
                            mode = 'Geofence'
                        if 740 < x1 < 1140:
                            cv2.rectangle(frame, (740, 300), (1140, 420), purple, 3)
                            mode = 'Customfence'
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == "Customfence":
            #x coordinate
            cv2.rectangle(frame, (0, 0), (210, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (10, 10), (60, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (20, 50), font, 3, red, 3)

            cv2.rectangle(frame, (80, 10), (130, 60), yellow, cv2.FILLED)
            cv2.putText(frame, str(custom_x), (95, 50), font, 2, red, 3)

            cv2.rectangle(frame, (150, 10), (200, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (160, 50), font, 3, red, 3)          
            
            #y coordinate
            cv2.rectangle(frame, (250, 0), (460, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (260, 10), (310, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (270, 50), font, 3, red, 3)

            cv2.rectangle(frame, (330, 10), (380, 60), yellow, cv2.FILLED)
            cv2.putText(frame, str(custom_y), (345, 50), font, 2, red, 3)

            cv2.rectangle(frame, (400, 10), (450, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (410, 50), font, 3, red, 3) 

            #add point
            cv2.rectangle(frame, (500, 0), (710, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (510, 10), (700, 60), yellow, cv2.FILLED)
            cv2.putText(frame, 'ADD POINT', (520, 50), font, 2, red, 2)

            #coordinates display
            cv2.rectangle(frame, (760, 0), (1000, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (770, 10), (820, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (780, 50), font, 3, red, 3)

            cv2.rectangle(frame, (840, 10), (920, 60), yellow, cv2.FILLED)
            if len(fence_coords) != 0:
                cur_point = str(fence_coords[custom_i])
            #cv2.putText(frame, cur_point, (850, 50), font, 2, red, 3)

            cv2.rectangle(frame, (940, 10), (990, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (950, 50), font, 3, red, 3)

            #remove point
            cv2.rectangle(frame, (1030, 0), (1280, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (1040, 10), (1270, 60), yellow, cv2.FILLED)
            cv2.putText(frame, 'REMOVE POINT', (1041, 50), font, 2, red, 2)

        if mode == "settracker":
            cv2.rectangle(frame, (0, 0), (210, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (10, 10), (60, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '-', (20, 55), font, 3, red, 3)

            cv2.rectangle(frame, (80, 10), (130, 60), yellow, cv2.FILLED)
            cv2.putText(frame, str(k), (90, 55), font, 2, red, 3)

            cv2.rectangle(frame, (150, 10), (200, 60), yellow, cv2.FILLED)
            cv2.putText(frame, '+', (160, 55), font, 3, red, 3)

            cv2.rectangle(frame, (1080, 0), (1280, 70), white, cv2.FILLED)
            cv2.rectangle(frame, (1090, 10), (1270, 60), yellow, cv2.FILLED)
            cv2.putText(frame, 'START', (1100, 55), font, 3, red, 3)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]
                fingers = handdetector.fingersUp()
                if fingers[1] == 1:
                    cv2.circle(frame, (x1, y1), 15, red, cv2.FILLED)
                    if 10 < y1 < 60:
                        if 10 < x1 < 60:
                            cv2.rectangle(frame, (10, 10), (60, 60), purple, 3)
                            k -= 1
                            if k < 0:
                                k = 0
                        if 140 < x1 < 190:
                            cv2.rectangle(frame, (140, 10), (190, 60), purple, 3)
                            k += 1
                            if k > 10:
                                k = 10
                        if 1090 < x1 < 1270:
                            cv2.rectangle(frame, (1090, 10), (1270, 60), purple, 3)
                            for i in range(k):
                                cv2.imshow('IMAGE', frame)
                                bbi = cv2.selectROI('IMAGE', frame)
                                tracker_i = TrDict['csrt']()
                                trackers.add(tracker_i, frame, bbi)
                            mode = 'Tracker'
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"
            #add instructions for tracker use on this page

        if mode == 'Tracker':
            begin = time.time()
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(a) for a in box]
                cv2.rectangle(frame, (x,y), (x+w,y+h), red, 2)
                pos_x = int((x + w)/2)
                pos_y = int((y + h)/2)
                a = str(pos_x)+" , "+str(pos_y)
                cv2.putText(frame, a, (x, y - 20), font, 1, red, 1)
                master_speed.append([pos_x,pos_y])

            if len(master_speed) > k*2:
                master_speed = master_speed[-(k*2):]

            last1 = master_speed[:k]
            last = master_speed[-k:]
            dist_list = []
            for i in range(k):
                dist_i = ((last[i][0] - last1[i][0])**2 + (last[i][1] - last1[i][1])**2)**0.5
                dist_list.append(dist_i)
            #print(dist_list)
            end = time.time()
            timediff = end - begin   
            #print(timediff)
            speedlist = []

            for i in range(k):
                speedlist.append(int(dist_list[i]/timediff))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
            flag = 0

            for box in boxes:
                for body in bodies:
                    for speed in speedlist:
                        (x, y, w, h) = [int(a) for a in box]
                        flag = check_overlap(box,body)
                        if speed == 0:
                            cv2.putText(frame,"Stillness detected", (x + 120, y - 20), font, 1, red, 2)
                        elif speed < 70 and speed > 0:
                            if flag:
                                cv2.putText(frame,"Walking detected", (x + 120, y - 20), font, 1, red, 2)
                                cv2.putText(frame, str(speed) + " pixels/second", (x, y + h + 20), font, 1, red, 2)

                        elif speed > 70 :
                            if flag:
                                cv2.putText(frame,"Running detected", (x + 120, y - 20), font, 1, red, 2)
                                cv2.putText(frame, str(speed) + " pixels/second", (x, y + h + 20), font, 1, red, 2)
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"
            
        if mode == 'Anomaly':
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == 'Violence':
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"

        if mode == 'Satellite':
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"
        
        if mode == "Vehicle":
            if len(lmList) != 0:
                fingers = handdetector.fingersUp()
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    mode = "Mainpage"


            #print(speedlist)
        if mode != 'Motion' and mode != 'Quadsplit':
            cv2.imshow('IMAGE', frame)
        cv2.waitKey(1)


