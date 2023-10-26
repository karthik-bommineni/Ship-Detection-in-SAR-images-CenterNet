import os
import shutil


def create_filtered_folder(folder_name, contents_list):
    try:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for index, content in enumerate(contents_list, 1):
            file_name = f"{folder_name}/file{index}.jpg"
            with open(file_name, 'w') as file:
                file.write(content)

        print(f'Folder {folder_name} created with contents successfully.')
    except Exception as e:
        print(f'An error has occurred: {e}')

def filtered_contents_list(xml_path_list):
    return xml_path_list

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

    for i in xml_list:
        xml_list_filename.append(xml_list[i].split('.')[0])

    jpg_list = os.listdir('jpg_path')
    jpg_list_filename = []

    for i in jpg_list:
        jpg_list_filename.append(jpg_list[i].split('.')[0])

    updated_jpg_list = []

    for i in xml_list_filename:
        for j in jpg_list_filename:
            if jpg_list_filename[j] == xml_list_filename[i]:
                updated_jpg_list.append(jpg_list_filename[j])

    
    copy_selected_contents('source path', 'destination path', updated_jpg_list)

    

    

    