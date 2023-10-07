'''
centernet + NMS版本 + 预测图
'''
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lim
"""
import os
import sys
import cfg

sys.path.append(r'./backbone')
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
# from resnet_dcn import ResNet
# from dlanet_dcn import DlaNet
from dlanet import DlaNet
from resnet import ResNet
from Loss import _gather_feat
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def draw_(filename, boxes, width=3, mode='xyxya'):
    '''
    filename: img_file_path
    result: [cx,cy,w,h,theta]
    '''

    img = Image.open(filename)
    w, h = img.size
    draw_obj = ImageDraw.Draw(img)

    for box in boxes:
        x_c, y_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (h, w), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=(0, 255, 0),
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=(0, 255, 0),
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=(0, 255, 0),
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=(0, 255, 0),
                      width=width)
    plt.imshow(img)
    plt.show()
    return img


# 绘制旋转椭圆
def draw_ellipse(filename, res, line_width=3):
    img = cv2.imread(filename)
    # w, h=img.size
    # draw_obj = ImageDraw.Draw(img)
    for class_name, lx, ly, rx, ry, ang, prob in res:
        result = [int((rx + lx) / 2), int((ry + ly) / 2), int(rx - lx), int(ry - ly), ang]
        # rect = ((int(lx), int(ly)), (int(rx), int(ry)), int(ang))
        # result=np.array(result)
        cx = int(result[0])
        cy = int(result[1])
        la = int(result[2] / 2)
        sa = int(result[3] / 2)
        img = cv2.ellipse(img, (cx, cy), (la, sa), int(ang), 0, 360, (0, 255, 0), thickness=line_width)
        # rect = ((x_c, y_c), (h, w), int(ang))
        # rect = ((x, y), (height, width), int(ang))
        # rect = cv2.boxPoints(rect)
        # rect = np.int0(rect)
        # draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
        #               fill=(0, 255, 0),
        #               width=line_width)
        # draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
        #               fill=(0, 255, 0),
        #               width=line_width)
        # draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
        #               fill=(0, 255, 0),
        #               width=line_width)
        # draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
        #               fill=(0, 255, 0),
        #               width=line_width)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow(img)
    # plt.show()

    img_save_path = cfg.RET_IMG + '/' + cfg.DATASET_NAME + '_' + cfg.Loss
    mkdir(img_save_path)
    img.save(os.path.join(img_save_path, os.path.split(filename)[-1]))


def draw(filename, result, line_width=3):
    img = Image.open(filename)
    w, h = img.size
    draw_obj = ImageDraw.Draw(img)
    for class_name, lx, ly, rx, ry, ang, prob in res:
        result = [int((rx + lx) / 2), int((ry + ly) / 2), int(rx - lx), int(ry - ly), ang]
        # rect = ((int(lx), int(ly)), (int(rx), int(ry)), int(ang))
        # result=np.array(result)
        x = int(result[0])
        y = int(result[1])
        height = int(result[2])
        width = int(result[3])

        # rect = ((x_c, y_c), (h, w), int(ang))
        rect = ((x, y), (height, width), int(ang))
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=(0, 255, 0),
                      width=line_width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=(0, 255, 0),
                      width=line_width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=(0, 255, 0),
                      width=line_width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=(0, 255, 0),
                      width=line_width)
        # anglePi = result[4]/180 * math.pi
        # anglePi = anglePi if anglePi <= math.pi else anglePi - math.pi

        # cosA = math.cos(anglePi)
        # sinA = math.sin(anglePi)
        #
        # x1=x-0.5*width
        # y1=y-0.5*height
        #
        # x0=x+0.5*width
        # y0=y1
        #
        # x2=x1
        # y2=y+0.5*height
        #
        # x3=x0
        # y3=y2
        #
        # x0n= (x0 -x)*cosA -(y0 - y)*sinA + x
        # y0n = (x0-x)*sinA + (y0 - y)*cosA + y
        #
        # x1n= (x1 -x)*cosA -(y1 - y)*sinA + x
        # y1n = (x1-x)*sinA + (y1 - y)*cosA + y
        #
        # x2n= (x2 -x)*cosA -(y2 - y)*sinA + x
        # y2n = (x2-x)*sinA + (y2 - y)*cosA + y
        #
        # x3n= (x3 -x)*cosA -(y3 - y)*sinA + x
        # y3n = (x3-x)*sinA + (y3 - y)*cosA + y
        #
        # draw.line([(x0n, y0n),(x1n, y1n)], fill=(0, 0, 255),width=5) # blue  横线
        # draw.line([(x1n, y1n),(x2n, y2n)], fill=(255, 0, 0),width=5) # red    竖线
        # draw.line([(x2n, y2n),(x3n, y3n)],fill= (0,0,255),width=5)
        # draw.line([(x0n, y0n), (x3n, y3n)],fill=(255,0,0),width=5)

    plt.imshow(img)
    plt.show()

    img_save_path = cfg.RET_IMG + '/' + cfg.DATASET_NAME + '_' + cfg.Loss
    mkdir(img_save_path)
    img.save(os.path.join(img_save_path, os.path.split(filename)[-1]))

########################################################################################################################
def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    mean = np.array(cfg.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.STD, dtype=np.float32).reshape(1, 1, 3)

    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):  # scores = heat图
    batch, cat, height, width = scores.size()  # b =1 , cate = 1, h = w = 128
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()  # [1,1,100] 找到在热图中xs和ys
    topk_xs = (topk_inds % width).int().float()  # [1,1,100]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)  # [1,100]
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()  # b = 1, c = 1, h=w=128
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)  # 执行一次max_pooling操作
    scores, inds, clses, ys, xs = _topk(heat, K=K)  # inds: index； clses = 0
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]  # [1,100,1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]  # [1,100,1]

    wh = _transpose_and_gather_feat(wh, inds)  # 根据inds在wh热图中提取出wh
    wh = wh.view(batch, K, 2)  # [1,100,2]

    ang = _transpose_and_gather_feat(ang, inds)  # 根据inds在ang热图中提取出ang
    ang = ang.view(batch, K, 1)  # [1,100,1]

    clses = clses.view(batch, K, 1).float()  # [[0]]
    scores = scores.view(batch, K, 1)
    # [cx,cy,h,w,ang] --> [lx, ly, rx, ry, ang]     转成coco格式
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2,
                        ang], dim=2)
    # bboxes = torch.cat([xs,ys,wh,ang],dim=2)

    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


# 模型预测
def process(images, return_time=False):
    with torch.no_grad():
        output = model(images)
        hm = output['hm'].sigmoid_()
        ang = output['ang'].relu_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)  # K 是最多保留几个目标
    if return_time:
        return output, dets, forward_time
    else:
        return output, dets


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


# 后处理预测结果，将预测结果映射回原图
def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))  # dets[i,:,0:2] = [100,2]--> [cx,cy]
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


# 后处理
def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = cfg.NUM_CLASSES  # @
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):  # j == 1
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = cfg.NUM_CLASSES  # @
    max_obj_per_img = 100  # @
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
        kth = len(scores) - max_obj_per_img
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2 + 1):
            keep_inds = (detections[j][:, 5] >= thresh)
            detections[j] = detections[j][keep_inds]
    return detections


def read_img(path):
    imgsets = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            path = dataset_img_path + line.strip() + '.' + cfg.IMG_EXT
            imgsets.append(path)
    return imgsets


def ellipse_nms(obbs, img_h, img_w, thr = 0.2):
    '''
    print('scores:',scores) # [?]
    print(scores.shape)
    print('classes:',classes)
    print(classes.shape)
    print('cxcy:',cxcy)     #[?,2]
    print(cxcy.shape)
    print('angels:',angels) #[?,1]    勿忘*180
    print(angels.shape)
    print('axies:',axies)   #[?,2]
    print(axies.shape)
    print('thr',thr)        #0.2
    '''
    order = [i for i in range(len(obbs))]
    classes, xylsa, scores = [],[],[]
    # 遍历每个obb
    for obb in obbs:
        assert len(obb) == 7
        cls_id, lx, ly, rx, ry, ang, prob = obb
        xylsa.append([int((rx + lx) / 2), int((ry + ly) / 2), int(rx - lx), int(ry - ly), ang])
        classes.append(cls_id)
        scores.append(prob)
    # 开启NMS
    keep=[]
    '''基本思路：循环每一个类别，抑制同一类别下的椭圆框'''
    while len(order)>0:
        # 假如只有一个框，则保存这个框然后结束循环
        if len(order) == 1:
            i=order[0]
            keep.append(i)
            break
        else:
            i=order[0]    #先保存了一个置信度最高的椭圆框的下标
            keep.append(i)
            order.remove(i)

            mask = np.zeros((img_h, img_w), np.uint8)  #

            # @
            # mask1 = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            # plt.imshow(mask1)
            # plt.show()

            # 提取分数最高的椭圆框信息
            cx=int(xylsa[i][0])
            cy=int(xylsa[i][1])
            long_axie=int(xylsa[i][2])
            short_axie=int(xylsa[i][3])
            angel=int(xylsa[i][4])
            ori_boxes= cv2.ellipse(mask,(cx,cy),(long_axie//2,short_axie//2),angel,0,360,(255,255,255),-1)
            areas1 = len(mask[mask == 255])

            # @
            # mask2 = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            # plt.imshow(mask2)
            # plt.show()

            # 循环当前类别下的框
            for index in order[0:]:            # 必须加上这个[0:],否则程序报错, 因为后续经过remove操作之后 遍历列表时会丢掉一个元素
                if classes[index]==classes[i]:
                    # 提取当前椭圆的信息
                    cx_ = int(xylsa[index][0])
                    cy_ = int(xylsa[index][1])
                    long_axie_ = int(xylsa[index][2])
                    short_axie_ = int(xylsa[index][3])
                    angel_ = int(xylsa[index][4])
                    # 在盖上去当前的椭圆
                    cv2.ellipse(mask, (cx_,cy_),(long_axie_ //2,short_axie_ //2),angel_, 0, 360, (90, 90, 90), -1)

                    # @
                    # mask3 = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                    # plt.imshow(mask3)
                    # plt.show()

                    areas2 = len(mask[mask == 90])
                    inter = areas1 - len(mask[mask == 255])
                    iou = inter / (areas1 + areas2 - inter + 1e-6)
                    #print('iou',iou)
                    # 如果交并比>0.2，此时这个框一定得删除！！！，否则程序不对。
                    if iou > thr:
                        order.remove(index)
                    '''
                    # 即使交并比<0.2,也不一定这个框就是正确的；比如000004.jpg的情形。所以，暂时不能立即将index 进行remove
                    else:
                        keep.append(index)
                        order.remove(index)
                    '''
                    # 记住把mask变回来，便于下次盖印。
                    mask = np.zeros((img_h, img_w), np.uint8)
                    cv2.ellipse(mask,(cx,cy),(long_axie //2,short_axie //2),angel,0,360,(255,255,255),-1)

                    # @
                    # mask4 = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                    # plt.imshow(mask4)
                    # plt.show()

                    if order == None:
                        break
                    #print('keep:',keep)

    return keep

if __name__ == '__main__':

    import cfg

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

    # 读取测试集的图像的list
    dataset_img_path = '/home/wujian/RCenterNet/data/' + cfg.DATASET_NAME + '/images/'
    test_txt = '/home/wujian/RCenterNet/data/' + cfg.DATASET_NAME + '/ImageSets/' + 'test.txt'
    imgsets = read_img(test_txt)
    # 此处测试图像的路径视情况需要改变。
    for image_name in imgsets:
        if image_name.split('.')[-1] == cfg.IMG_EXT:
            # if image_name.split('/')[-1] == '100000957.bmp':
            image = cv2.imread(image_name)
            img_h, img_w, _ = image.shape
            images, meta = pre_process(image)
            images = images.to(device)
            output, dets, forward_time = process(images,
                                                 return_time=True)  # dets: [1,100,7] --> [lx,ly,rx,ry,angle,score, class=0]

            dets = post_process(dets, meta)  # 后处理时候出现了负值？？[100,6] 少了class的维度。
            ret = merge_outputs(dets)  # ret == dets
            res = np.empty([1, 7])
            for i, c in ret.items():
                tmp_s = ret[i][ret[i][:, 5] > 0.3]  # [1,6] 无cls
                tmp_c = np.ones(len(tmp_s)) * i  # 类别多了+1的问题
                tmp = np.c_[tmp_c, tmp_s]
                res = np.append(res, tmp, axis=0)
            res = np.delete(res, 0, 0)
            res = res.tolist()
            print(res)                     # [cls_id, xmin, ymin, xmax, ymax, theta, score]
            # draw(image_name, res)        # 画旋转矩形
            # img_abs_path = dataset_img_path + image_name + cfg.IMG_EXT
            #draw_ellipse(image_name, res)  # 画旋转椭圆框

            # 对当前预测框拿去做NMS。此处得在原图上进行NMS。
            if not res:
                dets = []
                pass

            else:
                dets = []     # 保留去掉NMS的最终的检测结果
                keep = ellipse_nms(res,img_h, img_w)
                for i in keep:
                    dets.append(res[i])
                #res = res[ele for ele in keep]                      # 保留去掉NMS的res

            draw_ellipse(image_name, dets)  # 画旋转椭圆框

