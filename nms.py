import numpy as np


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]
    y2 = y1 + dets[:, 3]

    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1) * (y2 - y1)
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，argsort返回index存储在order中
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []

    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # x1左上角x坐标
        # y1左上角y坐标
        # x2右下角x坐标
        # y2右下角y坐标
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])

        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, dets[i, 2] + dets[order[1:], 2] - xx2 + xx1)
        h = np.maximum(0.0, dets[i, 3] + dets[order[1:], 3] - yy2 + yy1)
        inter = w * h
  
        # 交/并得到iou值
        union = areas[i] + areas[order[1:]] - inter
        ovr = inter / (areas[i] + areas[order[1:]] - inter)


        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep