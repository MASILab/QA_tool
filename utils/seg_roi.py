import pydicom

import nibabel as nib

import argparse

import numpy as np

def get_roi( img_path, mask_path, roi_path):

    img_nii = nib.load(img_path)
    img = img_nii.get_data()

    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_data()
    assert img.shape == mask.shape
 #   roi = np.zeros(mask.shape, dtype = np.uint8)
    x_list, y_list, z_list = [], [], []  
    for i in range(mask.shape[0]):
        if np.sum(mask[i, :, :]) > 20:
            x_list.append(i)
    for i in range(mask.shape[1]):
        if np.sum(mask[:, i, :]) > 20:
            y_list.append(i)
    for i in range(mask.shape[2]):
        if np.sum(mask[:, :, i]) > 20:
            z_list.append(i)
            #roi[:, :, i] = 1
    x_begin, x_end = x_list[0] - int(0.1 * len(x_list)), x_list[-1] + int (0.1 * len(x_list))
    y_begin, y_end = y_list[0] - int(0.1 * len(y_list)), y_list[-1] + int (0.1 * len(y_list))
    z_begin, z_end = z_list[0] - int(0.1 * len(z_list)), z_list[-1] + int (0.1 * len(x_list))
    print (x_begin, x_end, y_begin, y_end, z_begin, z_end)
    x_begin = max(0, x_begin)
    y_begin = max(0, y_begin)
    z_begin = max(0, z_begin)
    roi = img[x_begin: x_end, y_begin: y_end, z_begin: z_end]
    roi_nii = nib.Nifti1Image(roi, img_nii.affine, img_nii.header)
    nib.save(roi_nii, roi_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help='The original data path that want to segment')
    parser.add_argument('--mask',help='The original data path that want to segment')
    parser.add_argument('--roi',help='out path of the segmented image')
    args = parser.parse_args()
    get_roi(args.img, args.mask, args.roi)