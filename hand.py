import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, ais=cap.read()
    aisRGB=cv.cvtColor(ais,cv.COLOR_BGR2RGB)
    result=hands.process(aisRGB)
    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            for id,lm in enumerate(hlm.landmark):
                h,w,c=ais.shape
                cx, cy=int(lm.x * w),int(lm.y * h)
                print(id,cx,cy)
                if id==4: #enter any number from 0 to 20 for marking the point in hand(land mark)
                  cv.circle (ais, (cx , cy), 10 , (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(ais, hlm, mpHands.HAND_CONNECTIONS)

   
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(ais,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),5)    

    cv.imshow('photo',ais)
    cv.waitKey(1)


