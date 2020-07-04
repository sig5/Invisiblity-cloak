import cv2
import numpy as np
import time

vid = cv2.VideoCapture(0)
writer=cv2.VideoWriter_fourcc(*'XVID')
output=cv2.VideoWriter('output2.avi',writer,20.0,(640,480))
time.sleep(2)
count = 0
bg = 0
for i in range(0, 60):
    returnval, bg = vid.read()
    if not returnval:
        continue
bg = np.flip(bg, axis=1)
bg = np.flip(bg, axis=1)

while vid.isOpened():
    returnval, img = vid.read()
    if not returnval:
        break
    img=np.flip(img,axis=1)
    img=np.flip(img,axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 125, 50])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask1=mask1+mask2
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    mask2=cv2.bitwise_not(mask1)
    res1=cv2.bitwise_and(img,img,mask=mask2)
    res2=cv2.bitwise_and(bg,bg,mask=mask1)
    final=cv2.addWeighted(res1,1,res2,1,0)
    output.write(final)
    cv2.imshow("invisible",final)
    cv2.waitKey(1)
vid.release()
output.release()
cv2.destroyAllWindows()
