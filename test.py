from utils import find_dcm_folders

root = "../PET_NORMAL"
folders_with_dcm = find_dcm_folders(root)
for folder in folders_with_dcm:
    print(folder)
