import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from tqdm import tqdm
from utils import find_pt_lung_cancer_folders 

class DraggableHLine:
    def __init__(self, line):
        self.line = line
        self.press = None
        self.cid_press   = line.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_release = line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion  = line.figure.canvas.mpl_connect('motion_notify_event',   self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        contains, _ = self.line.contains(event)
        if not contains: return
        self.press = self.line.get_ydata()[0], event.ydata

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.line.axes: return
        y0, ypress = self.press
        new_y = y0 + (event.ydata - ypress)
        ymin, ymax = self.line.axes.get_ylim()
        new_y = np.clip(new_y, ymin, ymax)
        self.line.set_ydata([new_y, new_y])
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = None

def segment_with_sliders(folder: str, slice_axis: int = 2):
    names = ['CT.nii.gz', 'PET.nii.gz', 'SEG.nii.gz']
    paths = [os.path.join(folder, n) for n in names]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    ct_img  = nib.load(paths[0])
    ct_data = ct_img.get_fdata()
    zdim    = ct_data.shape[slice_axis]

    mid    = ct_data.shape[1] // 2
    cor    = np.take(ct_data, indices=mid, axis=1)
    disp   = cor.T
    zdim_d = disp.shape[0]

    y1 = 0.40 * zdim_d
    y2 = 0.80 * zdim_d

    fig, ax = plt.subplots(figsize=(6,6))
    plt.subplots_adjust(left=0.15, bottom=0.25)
    ax.imshow(disp, cmap='gray', origin='lower')
    ax.set_title("Drag lines or use sliders → click SAVE")

    line1 = ax.axhline(y1, color='red',   linewidth=2, label='start')
    line2 = ax.axhline(y2, color='blue',  linewidth=2, label='end')
    DraggableHLine(line1)
    DraggableHLine(line2)

    ax_s = fig.add_axes([0.15, 0.15, 0.7, 0.03])
    ax_e = fig.add_axes([0.15, 0.10, 0.7, 0.03])
    sl_s = Slider(ax_s, 'Start slice', valmin=0, valmax=zdim_d, valinit=y1, valfmt='%0.0f')
    sl_e = Slider(ax_e, 'End slice',   valmin=0, valmax=zdim_d, valinit=y2, valfmt='%0.0f')
    sl_s.on_changed(lambda v: (line1.set_ydata([v,v]), fig.canvas.draw_idle()))
    sl_e.on_changed(lambda v: (line2.set_ydata([v,v]), fig.canvas.draw_idle()))

    btn_ax = fig.add_axes([0.8, 0.025, 0.15, 0.05])
    btn = Button(btn_ax, 'Save', hovercolor='0.975')

    def on_save(event):
        ys = sorted([line1.get_ydata()[0], line2.get_ydata()[0]])
        start_idx = int(round(ys[0]))
        end_idx   = int(round(ys[1]))
        pct_start = start_idx / float(zdim_d)
        pct_end   = end_idx   / float(zdim_d)

        for in_path in tqdm(paths, desc=f"Saving segmented ({os.path.basename(folder)})"):
            img   = nib.load(in_path)
            data  = img.get_fdata()
            zdim_ = data.shape[slice_axis]
            i0 = int(round(pct_start * zdim_))
            i1 = int(round(pct_end   * zdim_))
            slicer = [slice(None)] * data.ndim
            slicer[slice_axis] = slice(i0, i1)
            cropped   = data[tuple(slicer)]
            out_img   = nib.Nifti1Image(cropped, img.affine, img.header)

            base, ext = os.path.splitext(os.path.basename(in_path))
            if ext == '.gz':
                base, _ = os.path.splitext(base)
                ext = '.nii.gz'
            out_name = f"{base}_segmented{ext}"
            out_path = os.path.join(folder, out_name)
            nib.save(out_img, out_path)

        plt.close(fig)

    btn.on_clicked(on_save)
    plt.show()

if __name__ == "__main__":
    csv_path = "./fb_metadata.csv"
    root_folder = r"F:\PET_FULL_BODY_DATASET\nifti\FDG-PET-CT-Lesions"
    folders = find_pt_lung_cancer_folders(csv_path, root_folder, skip_segmented=True, diagnosis = "NEGATIVE")

    for folder in tqdm(folders, desc="Processing folders"):
        segment_with_sliders(folder)