import os, random, shutil
from tqdm import tqdm
from KV_Utils.file_utils import img_name_helper
import pathlib
import project_params

def get_class_names(base_folder_path):
    data_dir = pathlib.Path(base_folder_path)
    result = [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]
    result.sort()
    return result

def list_error(check_list ,name_folder=''):
    if len(check_list) == 0:
        print('Error in Listing = ', name_folder, ' Exit Program')
        exit(-1)

def create_dataset_structure():
    Project_Name = project_params.Project_Name
    Base_Path = project_params.Base_Path
    train_dir = os.path.join(Base_Path, Project_Name +'_Data', 'train')
    validation_dir = os.path.join(Base_Path, Project_Name + '_Data', 'validation')
    img_name_helper.create_clean_dirs(train_dir)
    img_name_helper.create_clean_dirs(validation_dir)

    Unstructred_Data_Path = project_params.Unstructred_Data_Path
    class_names = get_class_names(Unstructred_Data_Path)
    list_error(class_names, Unstructred_Data_Path)

    for class_name in class_names:
        print('Procseeing Class = ', class_name, '\n')
        new_folder_path_train = os.path.join(train_dir, class_name)
        img_name_helper.create_clean_dirs(new_folder_path_train)
        new_folder_path_validation = os.path.join(validation_dir, class_name)
        img_name_helper.create_clean_dirs(new_folder_path_validation)

        old_class_path = os.path.join(Unstructred_Data_Path, class_name)
        file_list = img_name_helper.get_img_list(old_class_path, 'png')
        list_error(file_list, old_class_path)
        random.shuffle(file_list)

        exp_validation_frac = project_params.exp_validation_frac

        train_len = int(len(file_list) * (1 - exp_validation_frac))
        validation_len = len(file_list) - train_len
        for ii in tqdm(range(train_len)):
            shutil.copy2(file_list[ii], new_folder_path_train)

        for ii in tqdm(range(validation_len)):
            shutil.copy2(file_list[ii], new_folder_path_validation)

create_dataset_structure()
