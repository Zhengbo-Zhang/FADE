import cv2
import os
import numpy as np
from PIL import Image


def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(
        key=lambda x: int(x.replace("yoloframe", "").split('.')[0]))  # 最好再看看图片顺序对不
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size  # 获得图片分辨率，im_dir文件夹下的图片分辨率需要一致

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') #opencv版本是2
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    for i in im_list:
        im_name = os.path.join(im_dir, i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
        # count+=1
        # if (count == 200):
        #     print(im_name)
        #     break
    videoWriter.release()
    print('finish')


if __name__ == '__main__':
    # im_dir = "C:\\Users\\ggbond\\PycharmProjects\\高空抛物\\result\\video_by_demo\\"  # 帧存放路径
    # video_dir = "C:\\Users\\ggbond\\PycharmProjects\\高空抛物\\result\\final_video\\test12_20_2.mp4"  # 合成视频存放的路径
    # fps = 100  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    # frame2video(im_dir, video_dir, fps)
    data_path_root = '/mnt/data1/ghh/yolov5/video'
    video_path_root = '/mnt/data1/ghh/yolov5/video'
    for num in [95,568,784]:
        num = str(num)
        video_dir = os.path.join(video_path_root, f'{num}.mp4')
        image_path_root = os.path.join(data_path_root, num)
        fps = 20  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
        if not os.path.exists(video_path_root):
            os.makedirs(video_path_root)
        frame2video(image_path_root, video_dir, fps)
    print('finish!!!!')