'''
用于测试生成的json文件是否正确。将五边形的坐标表转化成四点坐标。
'''
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import os
import numpy as np


def draw(filename, boxes, width =3, mode = 'xyxya'):
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
    return img
    # 显示图像
    #plt.imshow(img)
    #plt.show()
    # pil保存图像
    #img.save('/home/wujian/R-CenterNet/pil.bmp')
    # boxes = np.array(boxes)
    # if mode == 'xyxya':
    #     boxes[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] * 0.5
    #     boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
    # img_ = cv2.imread(filename)
    # for box in boxes:
    #     img_ = cv2.rectangle(img_,(box[0],box[1]),(box[2],box[3]),(255,255,255),3)
    # cv2.imwrite('/home/wujian/R-CenterNet/det1.bmp',img_)

    #image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    #cv2.imshow('test',image)
    #cv2.waitkey(0)
    #cv2.imwrite('/home/wujian/R-CenterNet/cv2.bmp',image)



if __name__ == '__main__':

    img_path =  '/home/wujian/RSparseRCNN/projects/SparseRCNN/datasets/HRSC2016/JPEGImages/100001272.bmp'     # [828,1035,3]  h,w,c
    save_root = '/home/wujian/RSparseRCNN/projects/SparseRCNN/result/HRSC2016/'
    rboxes = [[376,230,472,77,140],[544,392,397,43,144],[634,526,350,50,149],[684,607,346,52,141]]
    rboxes = np.array(rboxes)

    # [cx,cy,chang,duan] --> [xmin,ymin,xmax,ymax]
    # rboxes[:, 0:2] = rboxes[:, 0:2] - rboxes[:, 2:4] * 0.5
    # rboxes[:, 2:4] = rboxes[:, 0:2] + rboxes[:, 2:4]
    #
    # img = cv2.imread(img_path)
    # for box in rboxes:
    #     img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,255),2)
    # image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # plt.imshow(image)
    # plt.show()

    #rboxes[:, 0:4] = rboxes[:, 0:4].astype(np.float32)
    img = draw(img_path,rboxes)
    img.save(save_root + '4.bmp')