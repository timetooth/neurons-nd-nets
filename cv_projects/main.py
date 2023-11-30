import os
import numpy as np
import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp.drawing_styles = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret , frame = cap.read()

        # detections
        image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

        # rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(27, 3, 163), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                                         )


        cv.imshow('hand tracking',image)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()