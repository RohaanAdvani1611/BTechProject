from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

classifier = load_model(r'C:\Users\Rohaan\Desktop\New Folder\MY PROGRAMS\VADnew.h5')
crime_labels = ['Abuse','Arrest','Arson','Assualt','Burglary', 'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    labels = []
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # roi = gray.astype('float') / 255.0
    # roi = img_to_array(roi)
    # roi = np.expand_dims(roi, axis=0)
    prediction = classifier.predict(frame)
    label = crime_labels[prediction.argmax()]
    # label_position = (x, y - 10)
    # cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VAD', frame)
    cv2.waitKey(1)