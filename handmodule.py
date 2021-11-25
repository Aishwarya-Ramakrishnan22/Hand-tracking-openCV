import cv2 as cv
import mediapipe as mp
import time

class hd():
    def __init__(self,mode=False,max=4,handcon=0.5,trackcon=0.5):
        self.mode=mode
        self.max=max
        self.handcon=handcon
        self.trackcon=trackcon
        
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.max,self.handcon,self.trackcon)
        self.mpDraw=mp.solutions.drawing_utils
        

    def findhands(self,ais,draw=True):
        aisRGB=cv.cvtColor(ais,cv.COLOR_BGR2RGB)
        self.result=self.hands.process(aisRGB)
        if self.result.multi_hand_landmarks:
            for hlm in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(ais, hlm, self.mpHands.HAND_CONNECTIONS)

        return ais                    



                

    def findposition(self,ais, handno=0, draw=True):
        lmlist =[]
        if self.result.multi_hand_landmarks:
            hlm=self.result.multi_hand_landmarks[handno]
            for id,lm in enumerate(hlm.landmark):
                h,w,c=ais.shape
                cx, cy=int(lm.x * w),int(lm.y * h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])
                #if draw:
                if id==0: # enter 0-20 number to mark the tracking point in hand for highlighting the track point
                    cv.circle (ais, (cx , cy), 10 , (255, 0, 255), cv.FILLED)
        return lmlist
        
        
    
    



def main():

    pTime=0
    cTime=0
    cap = cv.VideoCapture(0)
    dector=hd()

    while True:
        success, ais=cap.read()
        ais=dector.findhands(ais)
        lmlist=dector.findposition(ais)
        if len(lmlist)!=0:
            print(lmlist[0]) #enter number 0 to 20 to track particular position

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv.putText(ais,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),5)

        cv.imshow('photo',ais)
        cv.waitKey(1)


if __name__=="__main__":
    main()





