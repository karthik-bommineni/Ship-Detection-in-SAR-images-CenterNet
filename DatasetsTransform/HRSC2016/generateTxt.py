import os
import numpy as np
import os
import xml.etree.ElementTree as ET

# 检查xml文件中是否含有对象
def read_xml(path):

    flag = 1

    tree = ET.parse(path)
    gt_boxes = []
    gt_classes = []
    num_instances = 0

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["cx", "cy", "h", "w", "angle"]]
        # if cls == 'car' or cls == 'bus' or cls == 'vehicle':
        #     cls = 'vehicle'
        # else:
        #     continue
        gt_boxes.append(bbox)
        #gt_classes.append(class_names.index(cls))  # 标签是从0开始的。没有算back_ground

    if len(gt_boxes) == 0:
        flag =0
        return flag
    return flag

# 写
def write_txt(srcpath,savepath):
    with open(savepath,'w') as f:
        for name in os.listdir(srcpath):
            flag = read_xml(srcpath + name)
            if flag == 1:
                name = name.split('.')[0] + '\n'          # 加上换行符
                f.write(name)
            else:
                pass
    f.close()
    print('Done!')
if __name__ == '__main__':

    # 读入文件夹的路径
    train_val_xml_path = '/home/wujian/data/dataset/HRSC2016/Train/voc_xmls/'
    test_xml_path = '/home/wujian/data/dataset/HRSC2016/Test/voc_xmls/'
    # 保存txt的路径
    #！！！千万别把路径写到xml文件夹内部了！！！
    write_trainval_path = '/home/wujian/data/dataset/HRSC2016/Train/trainval.txt'
    write_test_path = '/home/wujian/data/dataset/HRSC2016/Test/test.txt'

    write_txt(train_val_xml_path, write_trainval_path)
    write_txt(test_xml_path, write_test_path)

    '''
    多检查几遍，因为这一步出错了，后面全完了。
    '''