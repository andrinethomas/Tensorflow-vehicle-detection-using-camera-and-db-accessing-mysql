#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:46:35 2018

@author: tensorflow-cuda
"""
import os
import numpy as np		      # importing Numpy for use w/ OpenCV
import cv2                            # importing Python OpenCV
from datetime import datetime         # importing datetime for naming files w/ timestamp
import paracept as aad
def diffImg(t0, t1, t2):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

threshold = 5000000                    # Threshold for triggering "motion detection"
cam = cv2.VideoCapture('rtsp://admin:Kgisl@123@192.168.1.64/doc/page/preview.asp')             # Lets initialize capture on webcam

winName = "Movement Indicator"	      # comment to hide window
cv2.namedWindow(winName)              # comment to hide window

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')
count = 1 
while True:
  ret, frame = cam.read()	      # read from camera
  totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
  text = "threshold: " + str(totalDiff)				# make a text showing total diff.
  cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
  if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):
    dimg= cam.read()[1]
#    cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
    cv2.imwrite('image{}.jpg'.format(count), dimg)
    aad.accept_and_die(os.path.join('image{}.jpg'.format(count)))
    count+=1
  timeCheck = datetime.now().strftime('%Ss')
  # Read next image
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  cv2.imshow(winName, cv2.resize(frame,(1280,960)))
  
  key = cv2.waitKey(60)
  if key == 27:			# comment this 'if' to hide window
    cv2.destroyWindow(winName)
    break