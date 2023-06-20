#from __future__ import print_functionTtime
from turtle import back
import cv2 as cv
import argparse
import numpy as np
import os
import sys
import xml.dom.minidom as xmldom
# from nms import py_cpu_nms
import time as T
import re
import linecache
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import numpy as np

# import ipdb


parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)

parser.add_argument('--trainortest',
                    type=str,
                    default="test")

parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='MOG2')

parser.add_argument('--video',
                    type=str,
                    help='video address',
                    default="/mnt/data3/gaokongdataset/video/")

parser.add_argument('--dataset',
                    type=str,
                    help='picture and annotation',
                    default="/mnt/data3/gaokongdataset/dataset/")

parser.add_argument('--val',
                    type=str,
                    help='input video',
                    default="/mnt/data3/gaokongdataset/dataslipt/test.txt")

args = parser.parse_args()

def getbounds(image, line, framenum, visual = True):
    fgMask = backSub.apply(image)
    if visual:
        visual_file_root = os.path.join("/mnt/data4/gaokongdataset","{}".format(line))
        if not os.path.exists(visual_file_root):
            os.mkdir(visual_file_root)
        visual_file = os.path.join(visual_file_root,"maskframe{}.jpg".format(framenum))
        if framenum==1:
            fgMask = 255 - np.asarray(fgMask)  # 图像转矩阵 并反色
        cv.imwrite(visual_file, fgMask)

pattern = "video"
with open(args.val) as f1:
    for line in f1:
        line = re.sub(pattern, "", line.rstrip())
        input = args.video + "{}.mp4".format(line)
        filename = args.dataset + line

        framenum = 0

        if args.algo == 'MOG2':
            backSub = cv.createBackgroundSubtractorMOG2(300, 100, False)
        elif args.algo == 'GSOC':
            backSub = cv.bgsegm.createBackgroundSubtractorGSOC(300, 100)
        elif args.algo == 'LSBP':
            backSub = cv.bgsegm.createBackgroundSubtractorLSBP()
        elif args.algo == 'GMG':
            backSub = cv.bgsegm.createBackgroundSubtractorGMG()
            backSub.setNumFrames(5)
            backSub.setUpdateBackgroundModel(True)
        elif args.algo == 'CNT':
            backSub = cv.bgsegm.createBackgroundSubtractorCNT()
            backSub.setIsParallel(True)
            backSub.setUseHistory(True)
            backSub.setMinPixelStability(1)
            backSub.setMaxPixelStability(4)
        elif args.algo == 'MOG':
            backSub = cv.bgsegm.createBackgroundSubtractorMOG()
        elif args.algo == 'KNN':
            backSub = cv.createBackgroundSubtractorKNN(300, 100, False)
        else:
            print("Wrong algo")
            sys.exit()

        capture = cv.VideoCapture(input)
        while True:
            framenum = framenum + 1
            ret, frame = capture.read()
            if frame is None:
                break
            bounds = getbounds(frame, line, framenum)
        
        print("{} DOWN".format(line))