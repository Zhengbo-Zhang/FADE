import os
import glob
from shutil import copyfile
from tqdm import tqdm
from collections import defaultdict


def rename(images, root):
    new_root = "/mnt/data2/like/yft/"
    index = 70083
    for image in tqdm(images):
        if os.path.exists(root + f"labels/{image}.txt"):
            copyfile(root + f"images/{image}.jpg", new_root + f"images/{index:06d}.jpg")
            copyfile(root + f"labels/{image}.txt", new_root + f"labels/{index:06d}.txt")
            index += 1
        else:
            print(image)


def filter():
    root = "/mnt/data2/like/yft_bk/"
    labels = sorted(glob.glob(root + "labels/*"))
    remove_list = []
    for label in labels:
        remain = False
        with open(label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] != '3':
                    remain = True
                    break
        if not remain:
            with open(root + 'remove_list.txt', 'a') as f:
                f.write(label + ' ' + str(len(lines)) + '\r\n')


def clean():
    root = "/mnt/data2/like/yft_bk/"
    with open(root + 'remove_list.txt', 'r') as f:
        remove_list = f.readlines()
    remove_list = sorted([x[:-1].split() for x in remove_list], key=lambda x: [int(x[1]), x[0]])
    d = defaultdict(int)
    for path in tqdm(remove_list):
        num = int(path[1])
        if num < 10:
            os.remove(path[0])
            os.remove(path[0].replace('labels', 'images').replace('txt', 'jpg'))


if __name__ == '__main__':
    # root = "/mnt/data2/like/yft_raw/pedlar/"
    # images = sorted([x.split('/')[-1][:-4] for x in glob.glob(root + "images/*")])
    # rename(images, root)
    clean()
