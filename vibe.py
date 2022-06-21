from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import os
import xml.dom.minidom as xmldom
from nms import py_cpu_nms
import time as T
import re
import linecache

parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)
parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='VIBE')
parser.add_argument('--video',
                    type=str,
                    default="/mnt/data2/gaokongdataset/video/")
parser.add_argument('--dataset',
                    type=str,
                    default="/mnt/data2/gaokongdataset/dataset/")
parser.add_argument('--gse_size',
                    type=int,
                    help='getStructuringElement kernel size',
                    default='9')
parser.add_argument(
    '--val',
    type=str,
    help='input video',
    default="/mnt/data2/gaokongdataset/Annotation/weak_light_test.txt")
parser.add_argument('--time',
                    type=str,
                    help='beginning and ending',
                    default="/mnt/data2/gaokongdataset/start.txt")
parser.add_argument(
    '--result_txt',
    type=str,
    help="output txt",
    default=
    "/mnt/data1/zzb/gaokongpaowu/IOU_0.3_output/VIBE_weak_light_test.txt")

args = parser.parse_args()


class ViBe:

    def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16):
        self.defaultNbSamples = num_sam
        self.defaultReqMatches = min_match
        self.defaultRadius = radiu
        self.defaultSubsamplingFactor = rand_sam

        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        height, width = img.shape
        self.samples = np.zeros((self.defaultNbSamples, height, width),
                                dtype=np.uint8)

        ramoff_xy = np.random.randint(-1,
                                      2,
                                      size=(2, self.defaultNbSamples, height,
                                            width))

        xr_ = np.tile(np.arange(width), (height, 1))

        yr_ = np.tile(np.arange(height), (width, 1)).T

        xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_

        xyr_ = xyr_ + ramoff_xy

        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        xyr = xyr_.astype(int)
        self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]

    def ProcessFirstFrame(self, img):
        self.__buildNeighborArray(img)
        self.fgCount = np.zeros(img.shape)
        self.fgMask = np.zeros(img.shape)

    def Update(self, img):
        height, width = img.shape
        dist = np.abs(
            (self.samples.astype(float) - img.astype(float)).astype(int))
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = np.sum(dist, axis=0)

        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background

        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0

        fakeFG = self.fgCount > 50
        matches[fakeFG] = False

        upfactor = np.random.randint(self.defaultSubsamplingFactor,
                                     size=img.shape)
        upfactor[matches] = 100
        upSelfSamplesInd = np.where(upfactor == 0)
        upSelfSamplesPosition = np.random.randint(
            self.defaultNbSamples, size=upSelfSamplesInd[0].shape)
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0],
                  upSelfSamplesInd[1])
        self.samples[samInd] = img[upSelfSamplesInd]

        upfactor = np.random.randint(self.defaultSubsamplingFactor,
                                     size=img.shape)
        upfactor[matches] = 100
        upNbSamplesInd = np.where(upfactor == 0)
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = np.random.randint(-1, 2, size=(2, nbnums))
        nbXY = np.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = img[upNbSamplesInd]

    def getFGMask(self):
        return self.fgMask


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


def getbounds(fgMask):
    #time_start=time.time()
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
        NEGATIVE = negative + NEGATIVE
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
FRAMENUM = 0

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
        framenum = 0

        capture = cv.VideoCapture(input)
        if capture.isOpened():
            framenum = framenum + 1
            rval, frame = capture.read()
        else:
            rval = False

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vibe = ViBe()
        vibe.ProcessFirstFrame(frame)

        while rval:
            rval, frame = capture.read()

            if frame is None:
                break
            framenum = framenum + 1
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            vibe.Update(gray)
            segMat = vibe.getFGMask()
            segMat = segMat.astype(np.uint8)
            bounds = getbounds(segMat)
            if len(bounds) > 0:
                timelist.append(framenum)
            precision(framenum, line, bounds)

        if len(timelist) == 0:
            begin = 0
            end = 0
        else:
            end = timelist[-1]
            begin = timelist[0]
        text = linecache.getline(args.time, int(line))
        time = []
        result = re.finditer(",", text)
        for i in result:
            time.append(i.span()[0])
        num = text[0:time[0]]
        if num != line:
            break
        gt_begin = int(text[time[0] + 1:time[1]])
        gt_end = int(text[time[1] + 1:-1])
        if end == 0:
            tro = 0
        elif begin > gt_end or end < gt_begin:
            tro = 0
        else:
            tro = (min(gt_end, end) - max(gt_begin, begin) +
                   1) / (max(gt_end, end) - min(gt_begin, begin) + 1)

        TRO = TRO + tro
        ALL = ALL + 1
        if (positive + negative) > 0:
            f.write(
                line +
                'precision is {:.2f} positive is {}  negative is {} tro{}\n'.
                format(positive /
                       (positive + negative), positive, negative, tro))
        else:
            f.write(num + 'notfind\n')
        print("{}done".format(line))
        end_time = T.time()
        FRAMENUM = FRAMENUM + framenum
        TIME = end_time - start_time + TIME
Precision = (POSITIVE / (POSITIVE + NEGATIVE))
Recall = (FIND / (FIND + NOTFIND))
F_measure = (2 * Precision * Recall) / (Precision + Recall)
f.write('%s:precision is %f\n' % (args.algo, Precision))
f.write('%s:recall is %f\n' % (args.algo, Recall))
f.write('%s:tro is %f\n' % (args.algo, (TRO / ALL)))
f.write('%s:fps is %f\n' % (args.algo, (FRAMENUM / TIME)))
f.write("%s:F-measure is %f\n" % (args.algo, F_measure))