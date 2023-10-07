'''
可视化脚本：
用于测试生成的xml文件是否正确。将五边形的坐标表转化成四点坐标。
'''
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import os
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import xml.dom.minidom
import numpy as np
import os
import math
import sys
import cv2

def draw(filename, boxes, width =5, mode = 'xyxya'):
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
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=(255, 0, 0),
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=(0, 0, 255),
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=(255, 0, 0),
                      width=width)
    # 显示图像
    plt.imshow(img)
    plt.show()
    return img

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


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 6],
           and has [cx,cy,h,w,theta, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    boxes_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        # if child_of_root.tag == 'size':
        #     for child_item in child_of_root:
        #         if child_item.tag == 'width':
        #             img_width = int(child_item.text)
        #         if child_item.tag == 'height':
        #             img_height = int(child_item.text)
        #
        # if child_of_root.tag == 'object':
        #     label = None
        #     for child_item in child_of_root:
        #         if child_item.tag == 'name':
        #             label = NAME_LABEL_MAP[child_item.text]
        #         if child_item.tag == 'bndbox':
        #             tmp_box = []
        #             for node in child_item:
        #                 tmp_box.append(float(node.text))
        #             assert label is not None, 'label is none, error'
        #             tmp_box.append(label)
        #             box_list.append(tmp_box)

        # ship
        # if child_of_root.tag == 'Img_SizeWidth':
        #     img_width = int(child_of_root.text)
        # if child_of_root.tag == 'Img_SizeHeight':
        #     img_height = int(child_of_root.text)
        if child_of_root.tag == 'object':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'bndbox':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'h':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'w':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'angle':
                            tmp_box[4] = float(node.text)

                    #tmp_box = coordinate_convert_r(tmp_box)
                        # assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    # if len(tmp_box) != 0:
                    boxes_list.append(tmp_box)
            #box_list = backward_convert(box_list)
            # print(box_list)
    gtbox_label = np.array(boxes_list, dtype=np.int32)

    return gtbox_label

if __name__ == '__main__':
    # 遍历src文件的名称，分别读取xml的文件，绘制图像，并将最终的结果保存进对应的文件夹中。
    root_path = "/home/wujian/sparse_rcnn_without_d2/projects/SparseRCNN/datasets/SSDD++/"
    img_path = root_path + 'JPEGImages/'
    xml_path = root_path + 'Annotations/'     # 上一步保存好转换后的路径
    save_path = root_path + 'Vision/'                           # 图像保存的路径， 前提在数据集中建立好Vision文件夹
    # 遍历文件夹下的每张图像
    for img_name in os.listdir(img_path):
        print(img_name)
        xml_name = img_name.split('.')[0] + '.xml'
        # 获取标签信息
        label_path = xml_path + xml_name
        rboxes = read_xml_gtbox_and_label(label_path)
        #print(rboxes)
        # 绘制图像
        img = draw(img_path + img_name,rboxes)
        img.save(save_path+img_name)

    #draw(img_path,rboxes)

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