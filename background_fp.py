from __future__ import print_function
from turtle import back
import cv2 as cv
import argparse
import numpy as np
import os
import xml.dom.minidom as xmldom
from nms import py_cpu_nms
import time as T
import sys
import re

parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)
parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='LSBP')
parser.add_argument('--gse_size',
                    type=int,
                    help='getStructuringElement kernel size',
                    default='9')
parser.add_argument('--video',
                    type=str,
                    default="/mnt/data2/gaokongdataset/background_video/")
parser.add_argument('--dataset',
                    type=str,
                    default="/mnt/data2/gaokongdataset/dataset/")
parser.add_argument(
    '--val',
    type=str,
    help='input video',
    default="/mnt/data2/gaokongdataset/Annotation/background_mp4.txt")
parser.add_argument('--time',
                    type=str,
                    help='beginning and ending',
                    default="/mnt/data2/gaokongdataset/start.txt")
parser.add_argument(
    '--result_txt',
    type=str,
    help="output txt",
    default=
    "/mnt/data1/zzb/gaokongpaowu/IOU_0.3_output/background_fp_mp4_LSBP_0.3.txt"
)
'''parser.add_argument('--output_frame',type=str,help="输出帧",default="/mnt/data1/zzb/gaokongpaowu/output_6_3_ps/")
parser.add_argument('--output_video',type=str,help="输出视频",default="/mnt/data1/zzb/gaokongpaowu/output_6_3_ps/")
'''

args = parser.parse_args()


def calculate(bound, mask):

    x, y, w, h = bound

    area = mask[y:y + h, x:x + w]

    pos = area > 0 + 0

    score = np.sum(pos) / (w * h)

    return score


def nms_cnts(cnts, mask, min_area):

    bounds = [cv.boundingRect(c) for c in cnts if cv.contourArea(c) > min_area]

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds]

    bounds = np.array(bounds)

    scores = np.expand_dims(np.array(scores), axis=-1)
    scores = np.array(scores)

    keep = py_cpu_nms(np.hstack([bounds, scores]), 0.3)

    return bounds[keep]


def IOU(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def getbounds(image):
    #time_start=time.time()
    fgMask = backSub.apply(image)
    line = cv.getStructuringElement(cv.MORPH_RECT,
                                    (args.gse_size, args.gse_size), (-1, -1))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, line)

    contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    bounds = nms_cnts(contours, fgMask, 5)

    deletelist = []
    final_bounds = []
    n = 0
    final_bounds = np.array(bounds)
    if final_bounds.size > 0:
        final_bounds[:, 2] = final_bounds[:, 0] + final_bounds[:, 2]
        final_bounds[:, 3] = final_bounds[:, 1] + final_bounds[:, 3]

    return final_bounds


def precision(framenum, line, box):
    global positive, POSITIVE, negative, NEGATIVE, iou_threshold, FIND, NOTFIND
    find = 0
    notfind = 0
    if not os.path.exists(args.dataset + line + '/Annotations/' +
                          "frame{}.xml".format(framenum)):
        boxA = [0, 0, 0, 0]
        negative = len(box)
        NEGATIVE = negative
    else:
        xml_file = xmldom.parse(args.dataset + line + '/Annotations/' +
                                "frame{}.xml".format(framenum))
        eles = xml_file.documentElement
        for i in range(len(eles.getElementsByTagName("xmin"))):
            xmin = eles.getElementsByTagName("xmin")[i].firstChild.data
            ymin = eles.getElementsByTagName("ymin")[i].firstChild.data
            xmax = eles.getElementsByTagName("xmax")[i].firstChild.data
            ymax = eles.getElementsByTagName("ymax")[i].firstChild.data
            boxA = [xmin, ymin, xmax, ymax]
            for boxB in box:
                if (len(boxA) and len(boxB)
                        and IOU(boxA, boxB) >= iou_threshold):
                    positive = positive + 1
                    POSITIVE = POSITIVE + 1
                    find = find + 1
                    FIND = FIND + 1
                else:
                    negative = negative + 1
                    NEGATIVE = NEGATIVE + 1
        notfind = len(eles.getElementsByTagName("xmin")) - find
        NOTFIND = NOTFIND + notfind


iou_threshold = 0.3
f = open(args.result_txt, 'a')
POSITIVE = 0.00
NEGATIVE = 0.00
ALL = 0.00
begin = 0
end = 0
tro = 0
FIND = 0
NOTFIND = 0
TRO = 0
TIME = 0
FRAMENUM = 0.0

pattern = "video"
with open(args.val) as f1:
    timelist = []
    for line in f1:
        start_time = T.time()
        line = re.sub(pattern, "", line.rstrip())
        input = args.video + "{}.mp4".format(line)
        filename = args.dataset + line

        positive = 0.00
        negative = 0.00
        framenum = 0.0

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

            bounds = getbounds(frame)
            if len(bounds) > 0:
                timelist.append(framenum)
            negative = negative + len(bounds)

        ALL = ALL + 1
        NEGATIVE = negative + NEGATIVE

        f.write(line +
                "  FRAMENUM: {}   NEGATIVE：{} \n".format(framenum, negative))
        print(line +
              "  FRAMENUM: {}   NEGATIVE：{}\n".format(framenum, negative))
        print("{}done".format(line))
        f.write('MEAN NEGATIVE {} \n'.format(negative / framenum))
        print(line + 'MEAN NEGATIVE {} \n'.format(negative / framenum))
        end_time = T.time()
        FRAMENUM = FRAMENUM + framenum
        TIME = end_time - start_time + TIME
f.write('%s: negative is %f\n' % (args.algo, NEGATIVE))
f.write('%s: mean negative is %f\n' % (args.algo, (NEGATIVE / FRAMENUM)))
f.write('%s: fps is%f\n' % (args.algo, (FRAMENUM / TIME)))
f.write('%s: framenum is %f\n' % (args.algo, FRAMENUM))
f.write('%s: time is %f\n' % (args.algo, TIME))
