#%%
import cv2
import numpy as np
import copy
import math

from keras.models import model_from_json  
import operator
from PIL import Image, ImageFilter

#%%
# parameters
threshold = 60  #  BINARY threshold
bgSubThreshold = 50
learningRate = 0
pred_thresh = 0.7

isBgCaptured = 0   # bool, whether the background captured


#%%
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def load_slr_model(folder,model_name):
    ''' Load Sign Language Recognition model
    Args:
        folder: model's folder
        model_name: name of the model
    '''
    json_file = open(folder + model_name + '.json','r')
    slr_json = json_file.read()
    json_file.close()
    print("Loaded model from disk")
    return model_from_json(slr_json)

def get_predict(model, img):
    ''' Get the predict character from slr model
    '''
    pred = model.predict(img)
    index, value = max(enumerate(pred[0]), key=operator.itemgetter(1))
    return [index, value]

#%%
# Open Camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# camera.set(10,200)
# cv2.namedWindow('trackbar')
# cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

#%%
labels = [chr(i) for i in range(ord('a'),ord('z')+1)]
labels.remove('j')
labels.remove('z')

folder = 'slr_models/'
model_name = 'slr_model_v2'
slr_model = load_slr_model(folder, model_name)

#%%
while camera.isOpened():
    ret, frame = camera.read()
    # threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
        # Right
    cv2.rectangle(frame, (frame.shape[1]-336,0),(frame.shape[1],336) ,(255,255,0), 2)
    cv2.imshow('camera', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        # img = frame

        # Right
        img = img[0:336,img.shape[1]-336:img.shape[1]]
        inp = Image.fromarray(np.array(img))
        inp = inp.filter(ImageFilter.FIND_EDGES)
        inp = np.array(inp)
        # inp1 = inp
            # Testing

        cv2.imshow('input 1', inp)
        inp = cv2.resize(inp,(28,28))
        inp = np.array([inp.flatten()])
        inp = inp.reshape(3,28,28,1)

        pred = get_predict(slr_model, inp)
        if pred[1] > pred_thresh :
            cv2.putText(img,'Predict: %s'%(labels[pred[0]]),(0,290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(img,'Percent: %.3f'%(pred[1]),(0,320), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        cv2.imshow('hand part', img)
        # cv2.imshow('input 2', inp2)
        # cv2.imshow('input main',inp_m)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
