# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:39:44 2023

@author: karth
"""

import numpy as np
import glob
import xml.etree.ElementTree as ET
import os

def short_edge_to_long_edge(angle, width, height):
    if width > height:
        # Bounding box is wider than tall
        long_edge_angle = angle + 90
    else:
        # Bounding box is taller than wide
        long_edge_angle = angle

    return long_edge_angle % 180  # Ensure the angle is in the range [0, 180)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

inp_path = 'C:\\Users\\karth\\OneDrive\\Desktop\\Rotate-CenterNet\\Ship-Detection-in-SAR-images-CenterNet\\data\\OSSDD\\annotations_train_xml\\'
out_path = 'C:\\Users\\karth\\OneDrive\\Desktop\\Rotate-CenterNet\\Ship-Detection-in-SAR-images-CenterNet\\data\\OSSDD\\annotations_train_xml_new\\'
mkdir(out_path)
ext = '*.xml'

files = glob.glob(inp_path + ext)

for file in files:
    file_name = os.path.basename(file)  # Use os.path.basename to get the file name
    print(file_name)
    
    tree = ET.parse(file)  # Use the full file path here
    root = tree.getroot()
    
    angles = root.findall('.//rotated_bbox_theta')
    widths = root.findall('.//rotated_bbox_w')
    heights = root.findall('.//rotated_bbox_h')
    
    for i in range(len(angles)):  # Corrected the loop range
        angle_value = float(angles[i].text)
        width_value = float(widths[i].text)
        height_value = float(heights[i].text)

        print('Creating a new XML file....')
        
        angles[i].text = str(short_edge_to_long_edge(angle_value, width_value, height_value))
    
    tree.write(os.path.join(out_path, file_name))  # Use os.path.join to create the output file path
