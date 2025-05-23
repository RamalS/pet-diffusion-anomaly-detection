@echo off
set folder_path_1=D:\dataset\PET_AD\ADNI\016_S_1263\PET_Brain\2007-03-27_11_26_53.0\I47396

python main.py example --example_folder="%folder_path_1%" --n_slices=5
