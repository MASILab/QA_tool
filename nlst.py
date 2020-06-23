from QA_tool import *

# fold_root = '/nfs/masi/NLST/ORI_DICOM/NLST_nocancer/NLST'
# save_dicom = '/nfs/masi/NLST/QA_results/20200513/noncancer/dicomQA.csv'

# save_length = '/nfs/masi/NLST/QA_results/20200513/noncancer/length.csv'

# #save_distance = '/nfs/masi/NLST/QA_results/20200513/noncancer/distanceQA.csv'

# #instanceN_fold_NLST(fold_root, save_dicom)

# #sliceDis_fold_NLST(fold_root, save_distance)

# #filter_few_slices(save_dicom)

# lenbody_fold_NLST(fold_root, save_length)

# check_affine_fold_NLST(fold_root = '/nfs/masi/NLST/nifti/NIFTI_cancer',
#                   save_csv_path = '/nfs/masi/NLST/QA_results/20200513/cancer/affine.csv')

#check_affine_fold_NLST(fold_root = '/nfs/masi/NLST/nifti/NIFTI_nocancer',
#                  save_csv_path = '/nfs/masi/NLST/QA_results/20200513/noncancer/affine.csv')

# max_res_fold_NLST(fold_root = '/nfs/masi/NLST/nifti/NIFTI_cancer',
#                   save_csv_path = '/nfs/masi/NLST/QA_results/20200513/cancer/resolution.csv')

# max_res_fold_NLST(fold_root = '/nfs/masi/NLST/nifti/NIFTI_nocancer',
#                   save_csv_path = '/nfs/masi/NLST/QA_results/20200513/noncancer/resolution.csv')

# ------------------ The follow for project ----------------------------- # 

# fold_root = '/nfs/masi/MCL/xnat/MCL'

# save_csv_path = '/nfs/masi/MCL/xnat/MCL_lenbody.csv'

# lenbody_fold(fold_root, save_csv_path)

lenbody_fold(fold_root = '/nfs/masi/SPORE/xnat/SPORE', save_csv_path = '/nfs/masi/SPORE/QA/SPORE_lenbody.csv')

# max_res_fold_NLST(fold_root = '/nfs/masi/SPORE/nifti/combine',
#                   save_csv_path = '/nfs/masi/SPORE/QA/resolution_comb0528.csv')

# check_affine_fold_NLST(fold_root = '/nfs/masi/SPORE/nifti/combine',
#                   save_csv_path = '/nfs/masi/SPORE/QA/affine_comb0528.csv')
