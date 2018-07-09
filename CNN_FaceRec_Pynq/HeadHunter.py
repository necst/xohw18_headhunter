#!/usr/bin/env python3.6 
# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import numpy as np
import six

import drawing
from hyperface import HyperFace
import log_initializer

from logging import getLogger, DEBUG, INFO
log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)


hyperface = None
def initialize():
    global hyperface
    hyperface = HyperFace(batchsize=10)

def runOnImage(img):
    img = img.astype(np.float32) / 255.0
    landmarks, visibilities, poses, genders, rects = hyperface(img)

    # Draw results
    for i in six.moves.xrange(len(landmarks)):
        landmark = landmarks[i]
        visibility = visibilities[i]
        pose = poses[i]
        gender = genders[i]
        rect = rects[i]

        landmark_color = (0, 1, 0)  # detected color
        drawing.draw_landmark(img, landmark, visibility, landmark_color, 0.5, denormalize_scale=False)
        drawing.draw_pose(img, pose, idx=i)

        gender = (gender > 0.5)
        drawing.draw_gender_rect(img, gender, rect)

    img *= 255
    return img


def run(frame=None, fromImg='Default.jpg', toImg='Result.jpg'):
    img = None
    
    if frame is not None:
        img = frame
    else:
        img = cv2.imread(fromImg)
    
    img = runOnImage(img)
    
    cv2.imwrite(toImg, img)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HyperFace CNN')
    parser.add_argument('--img', '-f', required=False, default=None, help='Image to analyze')
    args = parser.parse_args()

    logger.info('HyperFace Evaluation')
    initialize()

    if args.img:
        img = cv2.imread(args.img)
        imageName = args.img.replace("/", " ").split()[-1]
        logger.info('Feed model with image ' + imageName)

        img = runOnImage(img)

        cv2.imwrite(os.path.join('./', "res_" + imageName), img)
    else:
        logger.info('Feed model with live data from the webcam')

        videoIn = cv2.VideoCapture(0)
        maxRes = [1280,960]
        medRes = [640,480]
        lowRes = [320,240]
        camRes = medRes

        it = 0
        while True:
            logger.info('Taking video frame')
               
            videoIn.open(0)
            videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, camRes[0]);
            videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, camRes[1]);
            ret, frame = videoIn.read()
            videoIn.release()
            
            if ret:
                img = frame
                
                cv2.imwrite(os.path.join('./', "LiveCam_" + str(it) + ".jpg"), img)

                img = runOnImage(img)

                cv2.imwrite(os.path.join('./', "live_LiveCam" + str(it) + ".jpg"), img)
                
                it +=1
                
                


