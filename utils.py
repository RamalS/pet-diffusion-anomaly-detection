from pathlib import Path
from typing import List
import pandas as pd

def find_pt_lung_cancer_folders(
    csv_path: str,
    root_folder: str,
    skip_segmented: bool = False,
    diagnosis: str = "LUNG_CANCER",
    include_file_name: str = ""
) -> List[str]:
    df = pd.read_csv(csv_path, dtype=str)

    subj_ids = (
        df.loc[
            (df['Modality'] == 'PT') & 
            (df['diagnosis'] == diagnosis),
            'Subject ID'
        ]
        .dropna()
        .unique()
    )

    found_dirs = set()
    root = Path(root_folder)
    for sid in subj_ids:
        subject_dir = root / sid
        if not subject_dir.is_dir():
            continue

        for gz in subject_dir.rglob('*.gz'):
            folder = gz.parent

            if skip_segmented:
                if any(f for f in folder.iterdir() if 'segmented.nii.gz' in f.name):
                    continue

            if include_file_name:
                folder = folder / include_file_name

            found_dirs.add(str(folder))

    return sorted(found_dirs)


if __name__ == "__main__":
    csv_path = "./fb_metadata.csv"
    root_folder = r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions"
    folders = find_pt_lung_cancer_folders(csv_path, root_folder, diagnosis="LUNG_CANCER")
    for f in folders:
        print(f)

    print("done")
