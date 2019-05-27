#%%
# Import part
import cv2
import numpy as np
import copy
import math
import sys

from PIL import Image, ImageFilter

if len(sys.argv) < 2:
    print("Please input the label!\n Example:python data_gen.py a")
    exit()
label = sys.argv[1]

threshold = 60  #  BINARY threshold
bgSubThreshold = 50
learningRate = 0

isBgCaptured = 0   # bool, whether the background captured

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#%%
# Open Camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#%%
datas = []
count = 0
num_img = 200
blank_image = np.zeros((336,336))
temp_img = blank_image

#%%
while camera.isOpened() and count < num_img:
    ret, frame = camera.read()
    # threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
        # Right
    cv2.rectangle(frame, (frame.shape[1]-336,0),(frame.shape[1],336) ,(255,255,0), 2)
        # Left
    # cv2.rectangle(frame, (0,336),(336,0) ,(255,255,0), 2)
    cv2.imshow('camera', frame)

    temp_img1 = temp_img.copy()
    temp_img1 = cv2.putText(temp_img1,'Label: %s'%(label),(0,290), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    temp_img1 = cv2.putText(temp_img1,'Image_num: %d'%(count),(0,310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    cv2.imshow('Last Save image', temp_img1)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
      
        # Right Corner
        img = img[0:336,img.shape[1]-336:img.shape[1]]
        cv2.imshow('Trimed Imaged', img)

    # Keyboard OP
    k = cv2.waitKey(5)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # Remove last image
        if count > 0:
            if count == 1:
                datas = []
                temp_img = blank_image
            else:
                datas = datas[:-1]
                temp_img = datas[-1]
                count -= 1
            print ('!!!Remove last image!!!')
        else:
            print("!!!Blank memory!!!")
    elif k == ord('c'):   
        temp_img = img   
        datas.append(temp_img)
        count += 1
        print ('!!!Captured!!!')

#%%
print("Saving pictures")
for i in range(len(datas)):
    filename = 'output/'+ label + '%d.png'%(i+1)
    cv2.imwrite(filename,datas[i])