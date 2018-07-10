#import cv2
#import numpy as np
#
#
#
#cap = cv2.VideoCapture('rtsp://admin:kgisl@123@10.100.10.192/doc/page/preview.asp')
##cap = cv2.VideoCapture('http://192.168.1.64/doc/page/preview.asp')
##cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.88:554/web/admin.html')
#count = 1
#ret = True
#while (ret):
#    ret,img=cap.read()
#    cv2.imwrite('image{}.jpg'.format(count), img)
#    cv2.imshow('image', img)
#    count+=1
#    key = cv2.waitKey(10)
#    if key == 27:			# comment this 'if' to hide window
#        cv2.destroyAllWindows
#        cap.release
#        break

#import pyhik.hikvision

#camera = pyhik.hikvision.HikCamera('http://10.100.10.192', port=80, user='admin', pass='kgisl@123')

import cv2
#import paracept as pc
#import os
#vidcap = cv2.VideoCapture(0)
vidcap = cv2.VideoCapture('rtsp://admin:Kgisl@123@192.168.1.64/doc/page/preview.asp')
# just cue to 20 sec. position
count = 1
y=1
ret = True
while (ret):
#    vidcap.set(cv2.CAP_PROP_POS_MSEC,(1000))
    ret,image = vidcap.read()
    cv2.imshow('detection', image)
#    cv2.imwrite('image{}.jpg'.format(y), image)
#    path = os.path.join('image{}.jpg'.format(y))
#    pc.accept_and_die(path)
    
    y+=1
#    cv2.waitKey(10000)
        
#    cv2.waitKey(1000)
    
    
    
#    cv2.imshow('detection', image)
    
    cv2.waitKey(10)
#    if key == 27:			# comment this 'if' to hide window
#        cv2.destroyAllWindows
#        vidcap.release
#        break