#----------------------------------------------------#
#  Created by Riqiang Gao.                           #
#  Feel free to distribute at MASI lab.              #
#  contact: riqiang.gao@vanderbilt.edu               #
#----------------------------------------------------#

from QA_tool import *


#-------example 1: check the instance number of a single session ------------#

r1, r2, r3 = dcm_instance('/share2/gaor2/MCL/xnat_data/0327/MCL/363532318/363532318-20130610/new_max/DICOM')
#  r1 == r2 (that is r3 == 0) indicates this session pass the instance number check.
print (r1, r2, r3)


# ------example 2: check the slice distance of a single session ------------#
res = dcm_slicedistance('/share2/gaor2/MCL/xnat_data/0327/MCL/363532318/363532318-20130610/new_max/DICOM')
# if res == 1 indicates this session pass the slice distance check.


# ------example 3: convert a session from DICOM to NIFTI ------------#
dcm2nii(src_root = '/share2/gaor2/MCL/xnat_data/0327/MCL/363532318/363532318-20130610/new_max/DICOM', dst_root = '/share2/gaor2/MCL/xnat_data/0327/MCL/363532318/363532318-20130610/new_max/new_NIFTI')


# -----------example 4: check instance number a batch of sessions and generate QA report  -----------------------#
# note that if your folder structure is different from the example, you need small changes, it is very easy.     #
instanceN_fold(fold_root= '/share2/gaor2/MCL/xnat_data/0327/MCL', save_csv_path = '/share2/gaor2/MCL/xnat_data/0327/instanceNumberCheck.csv')


# ---example 5: filter some sessions with very limited slices (e.g. 1 or 2) but still can pass the instance number check---#
filter_few_slices(csv_path = '/share2/gaor2/MCL/xnat_data/0327/instanceNumberCheck.csv')


# -----------example 6: check slice distance a batch of sessions and generate QA report  -----------------------#
# note that if your folder structure is different from the example, you need small changes, it is very easy.     #
sliceDis_fold(fold_root= '/share2/gaor2/MCL/xnat_data/0327/MCL', save_csv_path = '/share2/gaor2/MCL/xnat_data/0327/sliceDistanceCheck.csv')


# -----------example 7: convert a batch of sessions from DICOM to NIFTI  -----------------------#
# note that if your folder structure is different from the example, you need small changes, it is very easy.   #
dcm2nii_project(SPORE_root = '/share2/gaor2/MCL/xnat_data/0327/MCL')
