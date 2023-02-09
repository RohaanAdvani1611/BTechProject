import cv2
import time
import PoseEstimatorModule as pm

cap = cv2.VideoCapture('Falltest.mp4')
detector = pm.poseDetector()
pos = 'S'
while True:
    succ, img = cap.read()
    img = cv2.resize(img, (1000, 600))
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    depth = detector.getDepth()
    print(depth)
    # cv2.putText(img, 'Fell Down', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    # print(lmList)
    if len(lmList) != 0:
        if lmList[0][2] < lmList[24][2] and lmList[24][2] < lmList[28][2] and lmList[0][2] < lmList[25][2] and lmList[24][2] < lmList[27][2]:
            cv2.putText(img, 'Standing', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            pos = 'S'
            begin = time.time()
        if lmList[0][2] > lmList[28][2] or lmList[0][2] > lmList[27][2]:
            cv2.putText(img, 'Lying Down', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            pos = 'L'
            end = time.time()
            if (end - begin < 2):
                cv2.putText(img, 'Fell Down', (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)