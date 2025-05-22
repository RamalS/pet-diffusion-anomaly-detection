import os
from typing import List

def find_dcm_folders(root_dir: str) -> List[str]:
    dcm_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fname.lower().endswith('.dcm') for fname in filenames):
            dcm_dirs.append(dirpath)
    return dcm_dirs
