import os

label_path_root = '/mnt/data1/ghh/paowu/labels'
target_path_root = '/mnt/data1/ghh/paowu/label'
files = os.listdir(label_path_root)
for file in files:
    label_file_path = os.path.join(label_path_root, file)
    with open(label_file_path, "r") as f:
        target_file_path = label_file_path = os.path.join(target_path_root, file)
        r = open(target_file_path, "w")
        content = f.readlines()
        for j, data_ in enumerate(content):
            data = list(data_)
            data[0]='1'
            data_=''.join(data)
            r.write(data_)
        r.close()
