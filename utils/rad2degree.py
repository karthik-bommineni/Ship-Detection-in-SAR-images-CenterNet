# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:09:59 2023

@author: karth
"""

import numpy as np
import glob
import xml.etree.ElementTree as ET

inp_path = ''
out_path = ''
ext = '*.xml'

files = glob.glob(inp_path + ext)

for file in files:
    file_name = file.split('\\')[-1]
    print(file)
    print(file_name)
    tree = ET.parse(file_name)
    root = tree.getroot()
    angles = root.findall('.//rotated_bbox_theta')
    
    for angle in angles:
        temp = angle.text
        angle.text = str(float(temp) * 180 / np.pi)
    tree.write(out_path + file_name)