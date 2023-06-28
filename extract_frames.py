'''
Script to extract frames and name them sequentially from a video

Takes in the args [--pathIn] and [--pathOut]
'''

import sys
import argparse
import os
import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    # Create the pathOut if it doesnt exist
    if not os.path.exists(pathOut):
        os.mkdir(pathOut)
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        # Uncomment this line to save 1 frame per 1000 frames
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        if success == False:
            break
        imPath = os.path.join(pathOut, 'frame{}.jpg'.format(count))
        cv2.imwrite(imPath, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)