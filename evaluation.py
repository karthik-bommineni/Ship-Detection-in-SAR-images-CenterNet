# -*- coding: utf-8 -*-
"""
未加NMS的评估代码
"""
import os
import cfg
import cv2
import math
import time
import torch
#import evaluation
import numpy as np
import sys
sys.path.append(r'./backbone')
#from resnet_dcn import ResNet
#from dlanet_dcn import DlaNet
from dlanet import DlaNet
from resnet import ResNet
import matplotlib.pyplot as plt
from predict import pre_process, ctdet_decode, post_process, merge_outputs
import time

# =============================================================================
# 推断
# =============================================================================
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def process(images, return_time=False):
    with torch.no_grad():
      t1 = time_sync()
      output = model(images)
      t2 = time_sync()
      hm = output['hm'].sigmoid_()
      ang = output['ang'].relu_()
      wh = output['wh']
      reg = output['reg']
      #torch.cuda.synchronize()
      #forward_time = time.time()
      forward_time = t2 - t1
      dets = ctdet_decode(hm, wh, ang, reg=reg, K=100) # K 是最多保留几个目标
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

# =============================================================================
# 常规 IOU
# =============================================================================
def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = bbox1[0] - bbox1[2] / 2.0, bbox1[1] - bbox1[3] / 2.0
        xmax1, ymax1 = bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0
        xmin2, ymin2 = bbox2[0] - bbox2[2] / 2.0, bbox2[1] - bbox2[3] / 2.0
        xmax2, ymax2 = bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1 ) 
    area2 = (xmax2 - xmin2 ) * (ymax2 - ymin2 )
 
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou
#bbox1 = [1,1,2,2]
#bbox2 = [2,2,2,2]
#ret = iou(bbox1,bbox2,True)
    


# =============================================================================
# 旋转 IOU
# =============================================================================
def iou_rotate_calculate(boxes1, boxes2):
#    print("####boxes2:", boxes1.shape)
#    print("####boxes2:", boxes2.shape)
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
#        print(int_area)
    else:
        ious=0
    return ious
# 用中心点坐标、长宽、旋转角
#boxes1 = np.array([1,1,2,2,0],dtype='float32')
#boxes2 = np.array([2,2,2,2,0],dtype='float32')
#ret = iou_rotate_calculate(boxes1,boxes2)
    


# =============================================================================
# 获得标签信息
# =============================================================================
def get_lab_ret(xml_path):    
    ret = []
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'angle':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                x1 = float(ob[0])
                y1 = float(ob[1])
                w = float(ob[2])
                h = float(ob[3])
                #angle = float(ob[4])*180/math.pi
                #angle = angle if angle < 180 else angle-180
                angle = float(ob[4])
                bbox = [x1, y1, w, h, angle]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret
    
def get_label_ret(xml_path):
    ret = []
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'name':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'angle':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                cls = ob[0]
                x1 = float(ob[1])
                y1 = float(ob[2])
                h = float(ob[3])
                w = float(ob[4])
                #angle = float(ob[4])*180/math.pi
                #angle = angle if angle < 180 else angle-180
                angle = float(ob[5])
                # 五点法 --> 八点法
                #x_c, y_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]
                rect = ((x1, y1), (h, w), angle)
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                rect_ = rect.flatten().tolist()
                bbox = [cls] + rect_
                ret.append(bbox)
                ob = []
                flag = 0
    return ret

def get_pre_ret(img_path, device):
    image = cv2.imread(img_path)
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)   # det[boxes, score, classes]

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)
    
    res = np.empty([1,7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:,5]>0.3]
        tmp_c = np.ones(len(tmp_s)) * (i)
        tmp = np.c_[tmp_c,tmp_s]
        res = np.append(res,tmp,axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()

    return res

def get_predict_ret(img_path, device):
    image = cv2.imread(img_path)
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)  # det[boxes, score, classes]
    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > 0.3]
        tmp_c = np.ones(len(tmp_s)) * (i)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()
    # 用来保存最终结果： [class, score, 8点]
    res_format = []
    for class_id, lx, ly, rx, ry, ang, prob in res:
        class_name = Classnames[class_id]
        #pre_one = np.array([(rx + lx) / 2, (ry + ly) / 2, rx - lx, ry - ly, ang])
        # 5 --> 8
        rect = (((rx + lx) / 2, (ry + ly) / 2), (rx - lx, ry - ly), ang)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        rect_ = rect.flatten().tolist()
        res_format.append([class_name] + [prob] + rect_)
    return res_format, forward_time
# 调用ap接口,或者说分别生成gt_txt和pred_txt文档。
def write_txt(path, content):
    with open(path,'w') as f:
        for line in content:
            # 写完一行在换行
            for ele in line:
                f.write(str(ele) + ' ')
            f.write('\n')

def write_gt_pred(imgsets, device):
    total_time = 0
    seen = 0
    for filename in imgsets:
        print(filename)
        img_path = os.path.join(dataset_img_path, filename + '.' + cfg.IMG_EXT)
        xml_path = os.path.join(dataset_xml_path, filename + '.xml')
        # 写入pred
        pre_ret, forward_time = get_predict_ret(img_path, device)   # pre_ret =[[class_name,lx,ly,rx,ry,ang, prob],]
        total_time += forward_time
        write_txt(save_pred_path + '/' + filename + '.txt', pre_ret)
        # 写入gt
        lab_ret = get_label_ret(xml_path)          # lab_res =[[cls, x1,y1, x2,y2, x3,y3, x4,y4],[cls, x1,y1, x2,y2, x3,y3, x4,y4]]
        write_txt(save_gt_path + '/' + filename + '.txt', lab_ret)
        seen +=1
    # 测试FPS
    fps = total_time / seen * 1000  # -->ms
    print("Write Done!")
    #print('fps:', fps)


def pre_recall(imgsets, device, iou=0.5):
    #imgs = os.listdir(root_path)
    num = 0
    all_pre_num = 0
    all_lab_num = 0
    miou = 0
    mang = 0
    for filename in imgsets:
        print(filename)
        img_path = os.path.join(dataset_img_path, filename + '.' + cfg.IMG_EXT)
        xml_path = os.path.join(dataset_xml_path, filename + '.xml')
        pre_ret = get_pre_ret(img_path, device)   # pre_ret = [[class_name,lx,ly,rx,ry,ang, prob],]
        lab_ret = get_lab_ret(xml_path)           # lab_res =[[cx, cy, w, h ,angle], [cx,cy, w, h, angle]]，

        all_pre_num += len(pre_ret)
        all_lab_num += len(lab_ret)
        for class_name,lx,ly,rx,ry,ang, prob in pre_ret:
            pre_one = np.array([(rx+lx)/2, (ry+ly)/2, rx-lx, ry-ly, ang])
            for cx, cy, h, w, ang_l in lab_ret:
                lab_one = np.array([cx, cy, h, w, ang_l])
                iou = iou_rotate_calculate(pre_one, lab_one)
                ang_err = abs(ang - ang_l)/180
                if iou > 0.5:
                    num += 1
                    miou += iou
                    mang += ang_err
    return num/all_pre_num, num/all_lab_num, mang/num, miou/num

def read_img(path):
    imgsets = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            path = line.strip()
            imgsets.append(path)
    return imgsets

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    # 确定类别的字典
    if cfg.DATASET_NAME == 'HRSC2016':
        Classnames = {1: 'ship'}
    elif cfg.DATASET_NAME == 'UCAS-AOD':
        Classnames = {1: 'plane', 2: 'car'}
    else:
        Classnames = { 1: 'dog', 2:'person',3: 'train', 4:'sofa', 5:'chair',
                      6:'car', 7:'pottedplant', 8:'diningtable', 9:'horse', 10:'cat',
                      11:'bus', 12:'bicycle', 13:'cow', 14:'motorbike', 15:'bird',
                      16:'tvmonitor', 17:'sheep', 18:'aeroplane', 19:'boat', 20:'bottle'}
    if cfg.NET == 'ResNet':
        model = ResNet(34)
        model.init_weights(pretrained=True)
    else:
        model = DlaNet(34)
    device = torch.device('cuda')
    best_path = './checkpoint/' + cfg.DATASET_NAME + '_' + cfg.Loss
    model.load_state_dict(torch.load(best_path + '/' + 'last.pth')['net'])
    model.eval()
    model.cuda()
    # 获取测试集图像的filename
    # 读取测试集的图像的list
    dataset_img_path = '/home/wujian/RCenterNet/data/' + cfg.DATASET_NAME + '/images/'
    dataset_xml_path = '/home/wujian/RCenterNet/data/' + cfg.DATASET_NAME + '/Annotations_xmls/'
    test_txt =  '/home/wujian/RCenterNet/data/' + cfg.DATASET_NAME + '/ImageSets/' + 'test.txt'
    # 评测结果保存地址
    save_dir = '/home/wujian/RCenterNet/mAP/test/' + cfg.DATASET_NAME + '_' + cfg.Loss
    save_gt_path =  '/home/wujian/RCenterNet/mAP/test/' + cfg.DATASET_NAME + '_' + cfg.Loss + '/' + 'ground-truth'
    save_pred_path = '/home/wujian/RCenterNet/mAP/test/' + cfg.DATASET_NAME + '_'+ cfg.Loss + '/' + 'detection-results'
    mkdir(save_gt_path)
    mkdir(save_pred_path)

    imgsets = read_img(test_txt)
    # 分别将gt和预测值写入对应的txt文档
    write_gt_pred(imgsets, device)

    # p, r, mang, miou = pre_recall(imgsets, device)
    # F1 = (2 * p * r) / (p + r)
    # print(p,r,F1)

    # 测试mAP
    from mAP.map_func import eval_mAP
    mAP = eval_mAP(save_dir)
    print('mAP:',mAP)
