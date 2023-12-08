"""
Created on Mon Nov 13 01:35:43 2023

@author: karthik
"""

########################### INTRODUCE #########################################

# Use it to create a small dataset from the existing dataset to run it on your 
# model to check whether or not it's working fine

# Use it to filter the images using the xml list as reference
# Comes in handy when you use tools like roLabelImg to annotate your images and 
# for some reason you were not able to annotate all the images and only want to
# select the images you annotated

###############################################################################

import os
import shutil

def copy_selected_contents(source_folder, destination_folder, selected_contents):
    try:
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        for item in selected_contents:
            source_path = os.path.join(source_folder, item) + ".jpg"
            destination_path = os.path.join(destination_folder, item) + ".jpg"

            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, destination_path)
                else:
                    shutil.copy2(source_path, destination_path)
                print(f'copied: {item}')

            else:
                print(f'Skipped: {item} (not found in source folder)')

        print(f"Selected contents from '{source_folder}' copied to '{destination_folder}' successfully.")

    except FileNotFoundError:
        print(f"Source folder '{source_folder}' not found.")
    except FileExistsError:
        print(f"Destination folder '{destination_folder}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    xml_list = os.listdir('xml_path')
    xml_list_filename = []
    print(xml_list)
    print('length of the xml_list is: ', len(xml_list))

    for i in xml_list:
        xml_list_filename.append(xml_list[i].split('.')[0])

    print(xml_list_filename)
    print('length of the xml_list_filename is: ', len(xml_list_filename))

    jpg_list = os.listdir('jpg_path')
    jpg_list_filename = []
    print(jpg_list)
    print('length of the jpg_list is: ', len(jpg_list))

    for i in range (0, len(jpg_list)):
        jpg_list_filename.append(jpg_list[i].split('.')[0])

    updated_jpg_list = []

    for i in range (0, len(xml_list_filename)):
        for j in jpg_list_filename:
            if jpg_list_filename[j] == xml_list_filename[i]:
                updated_jpg_list.append(jpg_list_filename[j])

    print(updated_jpg_list)
    print('length of the updated_jpg_list is: ', len(updated_jpg_list))

    
    copy_selected_contents('source path', 'destination path', updated_jpg_list)

    

    

    
