#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:20:45 2018

@author: opensource
"""

import glob
import cv2
import time
import os
import paracept as pc

while True:
    try:
        for (i,image_file) in enumerate(glob.iglob('/home/opensource/1970_01_06-1970_01_10/*.jpg')):
#           time.sleep(1)
#            print(os.path.isfile(image_file))
            img = cv2.imread(image_file)
#            cv2.imwrite('folder_processings/images/yo{}.JPG'.format(i), img)
#            os.remove(image_file)
#            process(img, i)
            im_name = os.path.basename(image_file)
            dandt = im_name[17:31]
            pc.accept_and_die(image_file, dandt)
    except OSError:
        time.sleep(1)
    

#print(os.path.isfile('images/*.JPG'))