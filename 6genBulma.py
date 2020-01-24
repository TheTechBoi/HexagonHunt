#CYMURGHSS <3

import numpy as np
import cv2
from networktables import NetworkTables


#NetworkTables.initialize(server='roborio-6025-frc.local') 
#table = NetworkTables.getTable("Vision") 

#cap = cv2.VideoCapture('altigenn.mp4') #video kaydı kullanırken

cap = cv2.VideoCapture(0)

cam_angle = 60
cam_width = 1280
cam_center = cam_width/2
cam_centerAngle = cam_angle/2
ratio = cam_angle / cam_width

hexagon_w = 76.2
hexagon_h = 76.2/2
hexagon_percentage = int((hexagon_h/hexagon_w)*100)
minHexW = 150
minHexH = 75
hexagonVerification = False

x_difference = 0
angle_difference = 0

#renkler
lower_color =  np.array([50, 100, 100])
upper_color =  np.array([70, 255, 255])

def maxPixel(liste, index):
    number = 0
    i = 0
    for i in range(len(liste)):
        if number < liste[i][0][index]:
            number = liste[i][0][index]

            

    return number

def minPixel(liste, index):
    b = 0
    number2 = liste[b][0][index]
    for b in range(len(liste)):
        if number2 > liste[b][0][index]:
            number2 = liste[b][0][index]
    return number2


if (cap.isOpened()== False): 
  print("Error opening video stream or file")

  
while cap.isOpened():
    _, frame = cap.read()
    


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((15,15),np.float32)/225
    
    smoothed = cv2.filter2D(hsv,-1,kernel)

    hsv_blur = cv2.medianBlur(smoothed,15)
    

    
    mask = cv2.inRange(hsv_blur, lower_color, upper_color)

    mask = cv2.erode(mask,kernel,iterations =2)
    mask = cv2.dilate(mask,kernel, iterations=2) 

    res = cv2.bitwise_and(frame,frame, mask = mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    
    #Yukarıda: Gerekli  format değiştirilmesi maskeleme blurlama vb.
    
    
    #Sınırları belirleme
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    #frame2 = frame.copy()
    #cv2.drawContours(frame2, cnts, -1 ,(0,255,0), 3) 

    if len(cnts) > 0:

            c = max(cnts,key =cv2.contourArea) #buyuk değerin seçimi
        
            maxX = maxPixel(c,0)
            minX = minPixel(c,0)
            maxY = maxPixel(c,1)
            minY = minPixel(c,1)
            
            shapeW = maxX - minX
            shapeH = maxY - minY

            if shapeW >= minHexW and shapeH >= minHexH:
                print('W is:')
                print(shapeW)
                print('H is:')
                print(shapeH)
                shape_percentage = int((shapeH/shapeW)*100)

                if (hexagon_percentage - 10) <= shape_percentage <= (hexagon_percentage + 10):
                    hexagonVerification = True
                else:
                    hexagonVerification = False


            if hexagonVerification == True:
                x = minX + int(shapeW/2) 
                y = minY + int(shapeH/2)
                
                cv2.circle(frame,(x,y), 6, (255, 0, 0), -1)
                cv2.circle(res,(x,y), 6, (255, 0, 0), -1)
                
                x_difference = cam_center - x
                angle_difference = 0-(x_difference*ratio)

            else:
                x = 0
                y = 0
                angle_difference = 0

    else:
        x = 0
        y = 0
        angle_difference = 0

    print("x : ")
    print(x)
    print("y : ")
    print(y) 
    print("Angle : ")
    print(angle_difference)

    #table.putNumber("X", x)
    #table.putNumber("Y", y)
    #table.putNumber("Angle",angle_difference)
    
        
    cv2.imshow('OHA',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break


        

    
    
      
 
cv2.destroyAllWindows() 
cap.release()
