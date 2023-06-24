import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

def convert(size, box):
    x = (box[0] + box[1]) / (size[0] * 2.0)
    y = (box[2] + box[3]) / (size[1] * 2.0)
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return abs(x), abs(y), abs(w), abs(h)


def xml2txt(ann_path, lab_path, images):
    if not os.path.exists(lab_path):
        os.mkdir(lab_path)
    for img_id in tqdm(images):
        classes = ["paowu"]
        in_file = open(os.path.join(ann_path, f'{img_id}.xml'), 'rb')
        out_file = open(os.path.join(lab_path, f'{img_id}.txt'), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            # if cls not in classes:
            #     continue
            # cls_id = classes.index(cls)
            cls_id = 0
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join('%.6f' % x for x in bb) + "\n")
            # print(f"{img_id}.jpg done")

def json2txt(ann_path, lab_path, images):
    if not os.path.exists(lab_path):
        os.mkdir(lab_path)
    for img_id in tqdm(images):
        classes = ["phone", "face", "hand"]
        in_file = open(os.path.join(ann_path, f'{img_id}.json'), 'rb')
        out_file = open(os.path.join(lab_path, f'{img_id}.txt'), 'w')
        str = in_file.read()
        tree = json.loads(str)
        root = tree
        w = int(root.get('imageWidth'))
        h = int(root.get('imageHeight'))
        objects_list = root.get('shapes')

        for obj in objects_list:
            cls = obj.get('label')
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            # cls_id = 0
            jsonpoint = obj.get('points')
            b = (float(jsonpoint[0][0]), float(jsonpoint[1][0]),
                 float(jsonpoint[0][1]), float(jsonpoint[1][1]))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join('%.6f' % x for x in bb) + "\n")
            # print(f"{img_id}.jpg done")


def labelvis(images):
    for img_id in tqdm(images):
        path = f"/mnt/data/like/phone/images/{img_id}.jpg"
        img = cv2.imread(path)
        if img is None:
            print(img_id)
            # img = Image.open(path)
            # img.seek(0)
            # img = img.convert('RGB')
            # img.save(f"/mnt/data/like/Data/smoking/images/{img_id}.jpg")
            continue
        img_h, img_w = img.shape[:2]
        with open(f"/mnt/data/like/phone/labels_corr/{img_id}.txt", "r") as f:
            labels = f.readlines()
            for label in labels:
                _, x, y, w, h = [float(x) for x in label.split()]
                leftTop = (int((x - (w / 2)) * img_w), int((y - (h / 2)) * img_h))
                rightBottom = (int((x + (w / 2)) * img_w), int((y + (h / 2)) * img_h))
                cv2.rectangle(img, leftTop, rightBottom, (0, 255, 0), 5)
        cv2.imwrite(f"/mnt/data/like/phone/gt_corr/{img_id}.jpg", img)


# def datasplit(images):
    # train, val = train_test_split(images, train_size=0.8, random_state=1998)
    # with open(f'/mnt/data1/ghh/phone/train.txt', 'w') as f:
    #     for id in tqdm(train):
    #         f.write(f"./images/{id}.jpg\n")
    # with open(f'/mnt/data1/ghh/phone/val.txt', 'w') as f:
    #     for id in tqdm(val):
    #         f.write(f"./images/{id}.jpg\n")
def datasplit():
    with open(f'/mnt/data1/ghh/paowuclass/test.txt', 'w') as f:
        for id in range(50854,65097):
            num = str(id).zfill(8)
            f.write(f"./images/{num}.jpg\n")

def filesplit(file_root, img_root, ann_root, files):
    if not os.path.exists(img_root):
        os.mkdir(img_root)
    if not os.path.exists(ann_root):
        os.mkdir(ann_root)
    for file in files:
        img_path = os.path.join(file_root, file + '.jpg')
        ann_path = os.path.join(file_root, file + '.txt')
        target_img_path = os.path.join(img_root, file + '.jpg')
        target_ann_path = os.path.join(ann_root, file + '.txt')
        os.system(f'cp {{}} {{}}'.format(img_path, target_img_path))
        os.system(f'cp {{}} {{}}'.format(ann_path, target_ann_path))

def val_devide(val_txt, target_img_path):
    path = val_txt
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{p} does not exist')
    im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
    for file in im_files:
        os.system(f'cp {{}} {{}}'.format(file, target_img_path))

if __name__ == '__main__':
    # img_root = '/mnt/data1/gaokongdataset/images'
    # file_list = sorted(x[:-4] for x in os.listdir(img_root))
    # images = sorted([x.split(os.sep)[-1][:-4] for x in os.path.join(img_root, '*') if x.split(os.sep)[-1][-4:] == '.jpg'])
    # annos = sorted([x.split('/')[-1][:-4] for x in glob.glob('/mnt/data/like/Data/smoking/annotations/*')])
    # labelvis(images)
    # datasplit(file_list)
    # xml2txt(ann_root, lab_root, images)
    # json2txt(ann_root, lab_root, images)
    # filesplit(file_root, img_root, ann_root, file_list)

    # annos = sorted([x.split('/')[-1][:-4] for x in glob.glob('/mnt/data1/ghh/paowuclass/jsons/*')])
    # ann_root = '/mnt/data1/ghh/paowuclass/jsons'
    # lab_root = '/mnt/data1/ghh/paowuclass/labels'
    # xml2txt(ann_root, lab_root, annos)

    datasplit()


    # image_root_path = '/mnt/data1/ghh/phone/images'
    # label_root_path = '/mnt/data1/ghh/phone/labels'
    # file_list = sorted(x[:-4] for x in os.listdir(image_root_path))
    # datasplit(file_list)

    # file_root_path = '/mnt/data1/ghh/smoke_image_guoneng'
    # annos = sorted(
    #      [x[:-4] for x in os.listdir(file_root_path) if x[-4:] == '.txt'])
    # image_root_path = '/mnt/data1/ghh/smoke_image_guoneng/images'
    # anans_root_path = '/mnt/data1/ghh/smoke_image_guoneng/labels'
    # filesplit(file_root_path, image_root_path, anans_root_path, annos)

    # file_root_path = '/mnt/data1/ghh/phone/json'
    # label_root_path = '/mnt/data1/ghh/phone/labels'
    # annos = sorted(
    #     [x[:-5] for x in os.listdir(file_root_path) if x[-5:] == '.json'])
    # json2txt(file_root_path, label_root_path, annos)