import cv2 as cv
import numpy as np
import os
import sys
import xml.dom.minidom as xmldom
import argparse
from nms import py_cpu_nms
import time as T
import re
import linecache
import matplotlib.pyplot as plt
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from util import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from pathlib import Path
import ipdb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)

parser.add_argument('--video',
                    type=str,
                    help='video address',
                    default="/mnt/data2/gaokongdataset/video/")
parser.add_argument('--trainortest',
                    type=str,
                    default="test")

parser.add_argument('--dataset',
                    type=str,
                    help='picture and annotation',
                    default="/mnt/data2/gaokongdataset/dataset/")

parser.add_argument('--val',
                    type=str,
                    help='input video',
                    default="/mnt/data2/gaokongdataset/dataslipt/iccv-visual.txt")

parser.add_argument('--scene',
                    type=str,
                    default="/mnt/data2/gaokongdataset/new_GT/scene.txt")

parser.add_argument('--Class',
                    type=str,
                    default="/mnt/data2/gaokongdataset/new_GT/class.txt")

parser.add_argument('--time',
                    type=str,
                    help='beginning and ending',
                    default="/mnt/data2/gaokongdataset/tro.txt")

parser.add_argument('--light',
                    type=str,
                    default="/mnt/data2/gaokongdataset/new_GT/light.txt")

parser.add_argument('--resolution',
                    type=str,
                    default="/mnt/data2/gaokongdataset/new_GT/resolution.txt")

parser.add_argument('--weather',
                    type=str,
                    default="/mnt/data2/gaokongdataset/new_GT/weather.txt")

parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='MOG2')

parser.add_argument('--result_txt',
                    type=str,
                    help="output address",
                    default="/mnt/data1/ghh/kalman-filter-in-single-object-tracking-main/class_my_yolo_f_b-0.3-10-last-timetest.txt")

parser.add_argument('--slice_height_num',
                    type=int,
                    default="2")

parser.add_argument('--slice_width_num',
                    type=int,
                    default="2") 

parser.add_argument('--ratio_overlap',
                    type=float,
                    default="0.2")                                       

parser.add_argument('--min_thre',
                    type=float,
                    default="0.2")     

parser.add_argument('--max_thre',
                    type=float,
                    default="0.2")
                        
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


#input formal[xmin, ymin, xmax, ymax]
# boxA是真值
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
    if (boxAArea + boxBArea - interArea)==0:
        return 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def getbounds(fgMask):
    #time_start=time.time()
    line = cv.getStructuringElement(cv.MORPH_RECT, (7, 7), (-1, -1))  # 返回指定形状和尺寸的结构
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
    # print(box)
    # print('\n')
    true_box = []
    global tp, TP, fp, FP, iou_threshold, FN, fn
    if not os.path.exists(args.dataset + line + '/Annotations/' +
                          "frame{}.xml".format(framenum)):
        boxA = [0, 0, 0, 0]
        fp = fp + len(box)
        FP = FP + len(box)
    else:
        xml_file = xmldom.parse(args.dataset + line + '/Annotations/' +
                                "frame{}.xml".format(framenum))
        eles = xml_file.documentElement
        gt_score = np.ones(len(eles.getElementsByTagName("xmin")))
        for boxB in box:
            score = 0
            for i in range(len(eles.getElementsByTagName("xmin"))):
                xmin = eles.getElementsByTagName("xmin")[i].firstChild.data
                ymin = eles.getElementsByTagName("ymin")[i].firstChild.data
                xmax = eles.getElementsByTagName("xmax")[i].firstChild.data
                ymax = eles.getElementsByTagName("ymax")[i].firstChild.data
                boxA = [xmin, ymin, xmax, ymax]
                if (len(boxA) and len(boxB)
                        and IOU(boxA, boxB) >= iou_threshold):
                    score = 1
                    gt_score[i] = 0
                    true_box.append(boxB)

            if score == 1:
                tp = tp + 1
                TP = TP + 1
            else:
                fp = fp + 1
                FP = FP + 1
        # if tp > len(eles.getElementsByTagName("xmin")):
        #     fn = 0
        # else:
        #     fn = len(eles.getElementsByTagName("xmin")) - tp
        fn = fn + sum(gt_score)
        FN = FN + sum(gt_score)
    return true_box

start_time = T.strftime('%Y-%m-%d_%H:%M:%S', T.localtime(T.time()))
dataset_root = "./data/labels"

f = open(args.result_txt, 'w')
iou_threshold = 0.3
TP = 0.00
FP = 0.00
ALL = 0.00
begin = 0
end = 0
tro = 0
FN = 0
TRO = 0
TIME = 0
FRAMENUM = 0
pattern = "video"
iou_thres = 0.3

# weights="/mnt/data1/ghh/yolov5/runs/train/exp18/weights/last.pt"
weights="/mnt/data1/ghh/yolov5/runs/train/exp29/last.pt"
data=ROOT / 'data/paowu.yaml'
imgsz=(960, 960)
conf_thres=0.4
box_color = (0,0,255) 

#加载yolo模型
device='0'
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
bs = 1
model.warmup(imgsz=(1 if pt else bs, 6, *imgsz))  # warmup

with open(args.val) as f1:
    statistics_TP = np.zeros((5, 17))
    statistics_FP = np.zeros((5, 17))
    statistics_FN = np.zeros((5, 17))
    statistics_TRO = np.zeros((5, 17))
    statistics_num = np.zeros((5, 17))
    for line in f1:
        timelist = []
        
        line = re.sub(pattern, "", line.rstrip())
        input = args.video + "{}.mp4".format(line)
        filename = args.dataset + line

        tp = 0.00
        fp = 0.00
        fn = 0.00
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
        initial = False

        # 使用yolov5获得初始位置
        start_time = T.time()
        while True:
            rval, frame = capture.read()
            if frame is None:
                break

            fgMask = backSub.apply(frame)
            fgMask = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)



            frame1=frame
            frame2=frame
            framenum = framenum + 1
            img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
            mask = letterbox(fgMask, imgsz, stride=stride, auto=True)[0]
            # Convert
            img = np.concatenate((img, mask), axis=2)
            mask = mask.transpose((2, 0, 1))[::-1]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            mask = np.ascontiguousarray(mask)
            img = np.ascontiguousarray(img)
            ma = mask
            im = img
            ma = torch.from_numpy(ma).to(device)
            im = torch.from_numpy(im).to(device)
            ma = ma.half() if model.fp16 else ma.float()
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            ma /= 255
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim 
                ma = ma[None]

            #example shape of img(torch.Size([1, 3, 384, 640]))
            # slice_img = []

            
            
            pred = model(im, ma, augment=False, visualize=False)
            # pred shape[1, x, 6]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=50)

            # import ipdb
            bounds = []
            for i, det in enumerate(pred):
                im0 = frame.copy()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                if len(det):
                    initial = True
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    print(det)
                    print("1")

                    for *xyxy, conf, cls in reversed(det):
                        bounds.append((torch.tensor(xyxy).view(1, 4)).view(-1).tolist())

            # for box in bounds:
            #     frame1 = cv.rectangle(frame1, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), color=box_color, thickness=3)
            #     visual_file_root = os.path.join("/mnt/data1/ghh/yolov5/visual","{}".format(line))
            #     if not os.path.exists(visual_file_root):
            #         os.mkdir(visual_file_root)
            #     visual_file = os.path.join(visual_file_root,"detframe{}.jpg".format(framenum))
            #     cv.imwrite(visual_file, frame1)

            true = precision(framenum, line, bounds)
            if tp > 0:
                timelist.append(framenum)
            if len(true):
                for true_box in true:
                    frame2 = cv.rectangle(frame2, (int(true_box[0]),int(true_box[1])), (int(true_box[2]),int(true_box[3])), color=box_color, thickness=3)
                visual_file_root = os.path.join("/mnt/data1/ghh/yolov5/video","{}".format(line))
                if not os.path.exists(visual_file_root):
                    os.mkdir(visual_file_root)
                visual_file = os.path.join(visual_file_root,"yoloframe{}.jpg".format(framenum))
                cv.imwrite(visual_file, frame2)
            else:
                visual_file_root = os.path.join("/mnt/data1/ghh/yolov5/video","{}".format(line))
                if not os.path.exists(visual_file_root):
                    os.mkdir(visual_file_root)
                visual_file = os.path.join(visual_file_root,"yoloframe{}.jpg".format(framenum))
                cv.imwrite(visual_file, frame)

            if tp > 0:
                timelist.append(framenum)

        if len(timelist) == 0:
            begin = 0
            end = 0
        else:
            end = timelist[-1]
            begin = timelist[0]
        text = linecache.getline(args.time, int(line))
        gt_time = []
        result = re.finditer(",", text)
        for i in result:
            gt_time.append(i.span()[0])
        num = text[0:gt_time[0]]
        if int(num) != int(line):
            break
        gt_begin = int(text[gt_time[0] + 1:gt_time[1]])
        gt_end = int(text[gt_time[1] + 1:-1])

        if end == 0:
            tro = 0
        elif begin > gt_end or end < gt_begin:
            tro = 0
        else:
            tro = (min(gt_end, end) - max(gt_begin, begin) +
                   1) / (max(gt_end, end) - min(gt_begin, begin) + 1)
        Class = int(linecache.getline(args.Class, int(line)).strip("\n")) - 1
        label = [Class]
        for i in range(1):
            j = label[i]
            statistics_TP[i][j] = statistics_TP[i][j] + tp
            statistics_FP[i][j] = statistics_FP[i][j] + fp
            statistics_FN[i][j] = statistics_FN[i][j] + fn
            statistics_TRO[i][j] = statistics_TRO[i][j] + tro
            statistics_num[i][j] = statistics_num[i][j] + 1
        TRO = TRO + tro
        ALL = ALL + 1

        f.write('class%d_tp%d_fp%d_fn%d\n' %(Class,tp,fp,fn))
        if (tp + fp) > 0:
            p = tp /(tp + fp)
            f.write(line +'precision is {:.6f}\n'.format(p))
        else:
            p = 0
            f.write(num + 'precision is 0\n')

        if (tp + fn) > 0:
            r = tp /(tp + fn)
            f.write(line +'recall is {:.6f}\n'.format(r))
        else:
            r = 0
            f.write(num + 'recall is 0\n')

        if (p + r) > 0:
            measure = 2 * p * r /(p + r)
            f.write(line +'F-measure is {:.6f}\n'.format(measure))
        else:
            f.write(num + 'F-measure is 0\n')
        f.write(line +'TRO is {:.6f}\n\n'.format(tro))
        print("{}done".format(line))
        end_time = T.time()
        FRAMENUM = FRAMENUM + framenum
        TIME = end_time - start_time + TIME

if ( TP + FP ) > 0:
    Precision = (TP / (TP + FP))
else:
    Precision = 0

if ( TP + FN ) > 0:
    Recall = (TP / (TP + FN))
else:
    Recall = 0

if Precision==0 and Recall==0:
    F_measure=0
else:
    F_measure = (2 * Precision * Recall) / (Precision + Recall)
f.write('%s:precision is %f\n' % (args.algo, Precision))
f.write('%s:recall is %f\n' % (args.algo, Recall))
f.write("%s:F-measure is %f\n" % (args.algo, F_measure))
f.write('%s:tro is %f\n' % (args.algo, (TRO / ALL)))
f.write('%s:fps is %f\n\n' % (args.algo, (FRAMENUM / TIME)))
statistics_P = np.zeros((5, 17))
statistics_R = np.zeros((5, 17))
statistics_F = np.zeros((5, 17))
statistics_T = np.zeros((5, 17))
for i in range(1):
    for j in range(17):
        if statistics_TP[i][j] + statistics_FP[i][j] == 0:
            P = 0
        else:
            P = (statistics_TP[i][j] / (statistics_TP[i][j] + statistics_FP[i][j]))
        statistics_P[i][j] = P
        f.write('%s:[%s][%s]precision of is %f\n' % (args.algo, i, j, P))

for i in range(1):
    for j in range(17):
        if statistics_TP[i][j] + statistics_FN[i][j] == 0:
            R = 0
        else:
            R = (statistics_TP[i][j] / (statistics_TP[i][j] + statistics_FN[i][j]))
        statistics_R[i][j] = R
        f.write('%s:[%s][%s]recall of is %f\n' % (args.algo, i, j, R))

for i in range(1):
    for j in range(17):
        if statistics_TP[i][j] + statistics_FN[i][j] == 0:
            R = 0
        else:
            R = (statistics_TP[i][j] / (statistics_TP[i][j] + statistics_FN[i][j]))
        if statistics_TP[i][j] + statistics_FP[i][j] == 0:
            P = 0
        else:
            P = (statistics_TP[i][j] / (statistics_TP[i][j] + statistics_FP[i][j]))
        if P + R == 0:
            F = 0
        else:
            F = (2 * P * R) / (P + R)
        statistics_F[i][j] = F
        f.write("%s:[%s][%s]F-measure of is %f\n" % (args.algo, i, j, F))

for i in range(1):
    for j in range(17):
        if statistics_num[i][j] == 0:
            T_1 = 0
        else:
            T_1 = statistics_TRO[i][j] / statistics_num[i][j]
        statistics_T[i][j] = T_1
        f.write('%s:[%s][%s]tro of is %f\n' % (args.algo, i, j, T_1))

end_time = T.strftime('%Y-%m-%d %H:%M:%S', T.localtime(T.time()))
