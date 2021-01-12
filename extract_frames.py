import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random as rnd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imgQnt = int(sys.argv[3])
def help_message():
   print("Usage: [Input_Video] [Output_Directory] [Frame_Count]")
help_message()


def detect_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale()
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def face_window(frame, window):
    x1,y1,x2,y2 = window
    roi = frame[y1:y1+x2, x1:x1+y2]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

# Camshift

def face_tracker(v, file_name):
    data_name = sys.argv[2] + file_name
    data = open(data_name,"w")

    total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    clip = rnd.randint(1500,3000)
    frame_seq = clip
    frame_no = (frame_seq/(total))
    v.set(1,frame_no)


    frameCounter = 0
    ret ,frame = v.read()
    if ret == False:
        return
    
    x1,y1,x2,y2 = detect_face(frame)
    pt = (0,x1+x2/2,y1+y2/2)
    data.write("%d,%d,%d\n" % pt) 
    frameCounter = frameCounter + 1

    track_window = (x1,y1,x2,y2)

    roi_hist = face_window(frame, (x1,y1,w,h)) 
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    cnt = clip
    while(1):
        ret ,frame = v.read() 
        cnt+=1
        if ret == False:
            break

        if cnt <= clip + imgQnt:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            x,y,w,h = track_window
            pt = (frameCounter,x+w/2,y+h/2)
            data.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = frame[y:y+h, x:x+w]

            croppedImg = roi_color
            if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=  0):
                frameCounter = frameCounter + 1
                continue
            print(croppedImg.shape)
            cv2.imshow('img',frame)
            cv2.imshow('img1',croppedImg)
            data_name = "./"+sys.argv[2]+str(frameCounter)+".jpg"
            cv2.imwrite(data_name, croppedImg)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            frameCounter = frameCounter + 1
        else:
            break

    data.close()


video = cv2.VideoCapture(sys.argv[1])
face_tracker(video, "data_camshift.txt")

