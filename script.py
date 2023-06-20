import os
from tqdm import tqdm

if __name__ == '__main__':
    root = '/mnt/data2/like/test/smoke_4/labels'
    files = os.listdir(root)
    for fname in tqdm(files):
        fpath = os.path.join(root, fname)
        new_labels = []
        with open(fpath, 'r') as f:
            labels = f.readlines()
            for label in labels:
                new_labels.append('0' + label[1:])
            with open(fpath, 'w+') as f:
                f.writelines(new_labels)
