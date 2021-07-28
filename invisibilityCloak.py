import cv2
import time
import numpy as np

# To save the output in a video form (in format of output.avi)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
outputFile=cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))

#to start the webcam
capture=cv2.VideoCapture(0)

# making the cam sleep for 2 secs
time.sleep(2)

#capturing background for 60 frames
bg=0

for i in range(60):
    ret,bg=capture.read()

# flipping the background
bg=np.flip(bg,axis=1)

# capturing the frame till the camera is open
while(capture.isOpened()):
    ret,image=capture.read()
    if not ret:
        break
    image=np.flip(image,axis=1)
    #Converting the color BGR to HSV(Hue saturation value)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #generating mask to detect red color
    lowerRed=np.array([0,120,50])
    upperRed=np.array([10,255,255])
    mask1=cv2.inRange(hsv,lowerRed,upperRed)
    lowerRed=np.array([170,120,70])
    upperRed=np.array([180,255,255])
    mask2=cv2.inRange(hsv,lowerRed,upperRed)
    mask1=mask1+mask2
    #open and expand the image in mask1
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    #selecting the area that does not have red color
    mask2=cv2.bitwise_not(mask1)
    #keeping the images without red color
    res1=cv2.bitwise_and(image,image,mask=mask2)
    #keeping the images with red color replaced with bg
    res2=cv2.bitwise_and(bg,bg,mask=mask1)
    #final output
    finalOutput=cv2.addWeighted(res1,1,res2,1,0)
    outputFile.write(finalOutput)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()