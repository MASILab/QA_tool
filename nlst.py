from QA_tool import *

fold_root = '/nfs/masi/NLST/ORI_DICOM/NLST_nocancer/NLST'
save_dicom = '/nfs/masi/NLST/QA_results/20200513/cancer/dicomQA.csv'

#save_distance = '/nfs/masi/NLST/QA_results/20200513/noncancer/distanceQA.csv'

#instanceN_fold_NLST(fold_root, save_dicom)

#sliceDis_fold_NLST(fold_root, save_distance)

filter_few_slices(save_dicom)
