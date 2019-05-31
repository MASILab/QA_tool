#----------------------------------------------------#
#  Created by Riqiang Gao.                           #
#  Feel free to distribute at MASI lab.              #
#  contact: riqiang.gao@vanderbilt.edu               #
#----------------------------------------------------#


import os
import pandas as pd
import nibabel as nib
import numpy as np
from skimage import transform, util
import re
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import shutil
import matplotlib.patches as patches
from glob import glob
import h5py
import pydicom

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def dcm2nii(src_root, dst_root):
    '''
    This function transfer the dcm form data to nii form
    :param src_root: The original data 's folder (root)
    :param dst_root: The target data 's folder
    :return:
    '''
    cmd_pre = "dcm2niix -m y "                                                    
    cmd = cmd_pre + "-o " + dst_root + " " + src_root
    os.system(cmd)
    print (cmd)
    
def dcm_instance(dcm_root):
    '''
    check a DICOM folder if it is valid
    '''
    dcm_list = os.listdir(dcm_root)
    slicePos = []
    instanceN = []
    for i in range(len(dcm_list)):
        ds = pydicom.dcmread(dcm_root + '/' + dcm_list[i])
        #slicePos.append(ds.SliceLocation)
        instanceN.append(ds[0x20, 0x13].value)
    print ('max and min of instanceN', max(instanceN), min(instanceN))
    return len(instanceN), max(instanceN) - min(instanceN) + 1,  max(instanceN) - min(instanceN) + 1 - len(instanceN)

def dcm_slicedistance(dcm_root):
    dcm_list = os.listdir(dcm_root)
    ds_list = []
    for i in range(len(dcm_list)):
        ds = pydicom.dcmread(dcm_root + '/' + dcm_list[i])
        ds_list.append(ds.SliceLocation)

    ds_sort = sorted(ds_list, reverse = True)
    #print (ds_sort)
    for i in range(0, len(ds_sort) - 2):
        print ((ds_sort[i] - ds_sort[i + 1]), (ds_sort[i + 1] - ds_sort[i + 2]))
        if not abs((ds_sort[i] - ds_sort[i + 1]) - (ds_sort[i + 1] - ds_sort[i + 2])) < (ds_sort[0] - ds_sort[1]):
            return 0
    return 1

def check_affine(nii_path):
    img_nib = nib.load(nii_path)
    aff_mat = img_nib.affine
    if aff_mat[0][0] < 0 and aff_mat[1][1] > 0 and aff_mat[2][2] > 0 and aff_mat[3][3] > 0:
        return 1
    else:
        return 0
    
def check_affine_fold(fold_root, save_csv_path): # different from sliceDis_fold and dcm_fold, this is for nifti folder.
    item_list = os.listdir(fold_root)
    res_list = []
    for i in range(len(item_list)):
        nib_paths = glob(fold_root + '/' + item_list[i] + '/*.nii.gz')
        assert len(nib_paths) == 1
        nib_path = nib_paths[0]
        res = check_affine(nib_path)
        res_list.append(res)
    data = pd.DataFrame()
    data['id'] = item_list
    data['affine_check'] = res_list
    data.to_csv(save_csv_path, index = False)
        
def sliceDis_fold(fold_root, save_csv_path):
    subj_list = os.listdir(fold_root)
    sess, single_folder, diff = [], [], []
    for i in range(0, len(subj_list)):
        #if i > 3: break
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            if len(instance_list) == 1:
                single_folder.append(1)
            else:
                single_folder.append(0)
#            size_list = []
#             for k in range(len(instance_list)):
#                 print (sess_path + '/' + instance_list[k])
#                 if os.path.exists(sess_path + '/' + instance_list[k] + '/secondary') and not os.path.exists(sess_path + '/' + instance_list[k] + '/DICOM'):
#                     os.rename(sess_path + '/' + instance_list[k] + '/secondary', sess_path + '/' + instance_list[k] + '/DICOM')
#                 size = len(os.listdir(sess_path + '/' + instance_list[k] + '/DICOM'))
#                 size_list.append(size)
            #max_index = size_list.index(max(size_list))
            #break
            #os.rename(sess_path + '/' + instance_list[max_index], sess_path + '/new_max')
            try:
                same = dcm_slicedistance(sess_path + '/new_max/DICOM')
                diff.append(same)
            except:
                diff.append('')
                print ('dicom error')
    data = pd.DataFrame()
    data['sess'] = sess
    data['single_folder'] = single_folder
    data['distance_check'] = diff
    data.to_csv(save_csv_path, index = False)
    
def instanceN_fold(fold_root, save_csv_path):
    subj_list = os.listdir(fold_root)
    sess, single_folder, instanceN, dicomN, diff = [], [], [], [],[]
    for i in range(0, len(subj_list)):
        #if i > 30: break
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            if len(instance_list) == 1:
                single_folder.append(1)
            else:
                single_folder.append(0)
            size_list = []
            for k in range(len(instance_list)):
                print (sess_path + '/' + instance_list[k])
                if os.path.exists(sess_path + '/' + instance_list[k] + '/secondary') and not os.path.exists(sess_path + '/' + instance_list[k] + '/DICOM'):
                    os.rename(sess_path + '/' + instance_list[k] + '/secondary', sess_path + '/' + instance_list[k] + '/DICOM')
                size = len(os.listdir(sess_path + '/' + instance_list[k] + '/DICOM'))
                size_list.append(size)
            max_index = size_list.index(max(size_list))
            #break
            os.rename(sess_path + '/' + instance_list[max_index], sess_path + '/new_max')
            try:
                inst_n, dicom_n, same = dcm_instance(sess_path + '/new_max/DICOM')
                instanceN.append(inst_n)
                dicomN.append(dicom_n)
                diff.append(same)
            except:
                instanceN.append('')
                dicomN.append('')
                diff.append('')
                print ('dicom error')
    data = pd.DataFrame()
    data['sess'] = sess
    data['single_folder'] = single_folder
    data['instanceN'] = instanceN
    data['dicomN'] = dicomN
    data['dicomN-instanceN'] = diff
    data.to_csv(save_csv_path, index = False)
    
    
def filter_few_slices(csv_path):
    df = pd.read_csv(csv_path)
    auto_QA_result = []
    for i, item in df.iterrows():
        if item['dicomN-instanceN'] > 0 or item['instanceN'] < 20:
            auto_QA_result.append('bad')
        else:
            auto_QA_result.append('good')
    df['auto'] = auto_QA_result
    df.to_csv(csv_path, index = False)

def dcm2nii_MCL(MCL_root):
    '''
    exp: dcm2nii_MCL('/share5/gaor2/data/MCL_subjects/MCL')
    :param MCL_root:
    :return:
    '''
    subj_list = os.listdir(MCL_root)
    for subj in subj_list:
        print (subj)
        subj_path = MCL_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for fder in fder_list:
                fder_path = sess_path +'/' + fder
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        
                        mkdir(fder_path + '/new_NIFTI')
                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')
                        
def dcm2nii_condition_MCL(MCL_root, txt_file):
    '''
    exp: dcm2nii_MCL('/share5/gaor2/data/MCL_subjects/MCL')
    :param MCL_root:
    :return:
    '''
    f = open(txt_file)
    lines = f.readlines()
    need_list = [line.strip() for line in lines]
    subj_list = os.listdir(MCL_root)
    for subj in subj_list:
        print (subj)
        subj_path = MCL_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            if sess not in need_list:
                continue
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for fder in fder_list:
                fder_path = sess_path +'/' + fder
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        mkdir(fder_path + '/new_NIFTI')
                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')                   

def dcm2nii_project(SPORE_root):  # this make sense for spore and mcl
    subj_list = os.listdir(SPORE_root)
    for i in range(len(subj_list)):
        subj = subj_list[i]
        print (i, len(subj_list), subj)
        #if (subj == '40593716009'): continue
        subj_path = SPORE_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for i in range(len(fder_list)):
                fder = fder_list[i]
                ori_fder_path = sess_path +'/' + fder
                fder_path = sess_path + '/file' + str(i)
                if not os.path.exists(fder_path):
                #print ('mv ' + ori_fder_path + ' ' + fder_path)
                    os.rename(ori_fder_path ,fder_path)
                    print (fder_path)
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        mkdir(fder_path + '/new_NIFTI')
                        if len(os.listdir(fder_path + '/new_NIFTI')) != 0:
                            continue
                        #print (fder_path + '/DICOM', fder_path + '/new_NIFTI')
                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')
                        
def dcm2nii_condition(SPORE_root, qa_csv):  # this make sense for spore and mcl, and introduce condition for dcm2nii
    subj_list = os.listdir(SPORE_root)
    df = pd.read_csv(qa_csv)
    df_qa = df.loc[df['dicomN-instanceN'] <= 0]
    #df_qa = df_qa.loc[df_qa['']]
    sess_qa = df_qa['sess'].tolist()
    for i in range(len(subj_list)):
        subj = subj_list[i]
        print (i, len(subj_list), subj)
        #if (subj == '40593716009'): continue
        subj_path = SPORE_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            if sess not in sess_qa:
                continue
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for i in range(len(fder_list)):
                fder = fder_list[i]
                ori_fder_path = sess_path +'/' + fder
                fder_path = sess_path + '/new_max'
                if not os.path.exists(fder_path):
                #print ('mv ' + ori_fder_path + ' ' + fder_path)
                    os.rename(ori_fder_path ,fder_path)
                    print (fder_path)
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        mkdir(fder_path + '/new_NIFTI')
                        if len(os.listdir(fder_path + '/new_NIFTI')) != 0:
                            continue
                        #print (fder_path + '/DICOM', fder_path + '/new_NIFTI')
                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')
                        
def cp_to_combine(data_root, new_root, QA_csv):  # this copy the QA data to combine. 
    df = pd.read_csv(QA_csv)
    for i, item in df.iterrows():
        if item['usability'] != item['usability']:
            ori_path = data_root + '/' + str(item['subject']) + '/' + str(item['time']) + '/' + str(item['niftiName']) 
            
            new_path = new_root + '/' + str(item['subject']) + 'time' + str(item['time'])
            mkdir(new_path)
            os.system('mv ' + ori_path + ' ' + new_path)
            print ('mv ' + ori_path + ' ' + new_path)
 
def get_trdata_project(data_root, new_root):
    '''
    I made some change compare with data_tool.py. This function for after the dicom header check.
    '''
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        subj_id = re.split('[-_]', subj_list[i])[0]
        #subj_id = subj_list[i].split('_')[1]
        mkdir(new_root + '/' + subj_id)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
#             if len(os.listdir(sess_path)) != 1:
#                 print (sess_path, 'len != 1')
#                 continue
            nifti_path = sess_path + '/file0/new_NIFTI'
            sess_id = re.split('[_-]', sess_list[j])[-1]
            
            
            data_list = glob(os.path.join(nifti_path, '*.nii.gz'))
            if len(data_list) == 0:
                continue
                
            mkdir(new_root + '/' + subj_id + '/' + sess_id)
            
            if (len(os.listdir(new_root + '/' + subj_id + '/' + sess_id)) > 0):
                print (new_root + '/' + subj_id + '/' + sess_id, ' already completed')
                continue
            
            os.rename(data_list[0], nifti_path + '/move.nii.gz')
            new_name = subj_id + 'time' + sess_id + '.nii.gz'
            print ('mv ' + nifti_path + '/move.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
            os.system('mv ' + nifti_path + '/move.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
            
def check_empty(data_root):
    '''
    Check the MCL_time fold if exist empty session
    '''
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
            nifti_list = glob(os.path.join(sess_path, '*.nii.gz'))
            print (nifti_list)
#             if len(nifti_list) != 1:
#                 print ('rm -r ' + sess_path)
#                 os.system('rm -r ' + sess_path)
            assert len(nifti_list) == 1
        
          
def combine_QA_csv(QA_paths, combine_path):
    sess_list, single_folder_list, instanceN_list, dicomN_list, DicomN_instanceN_list = [],[],[],[],[]
    path_list = []
    for path in QA_paths:
        df = pd.read_csv(path)
        sess_list += df['sess'].tolist()
        single_folder_list += df['single_folder'].tolist()
        instanceN_list += df['instanceN'].tolist()
        dicomN_list += df['dicomN'].tolist()
        DicomN_instanceN_list += df['dicomN-instanceN'].tolist()
        path_list += [path] * len(df)
    data = pd.DataFrame()
    data['sess'] = sess_list
    data['single_folder'] = single_folder_list
    data['instanceN'] = instanceN_list
    data['dicomN'] = dicomN_list
    data['dicomN-instanceN'] = DicomN_instanceN_list
    data['path'] = path_list
    
    data.to_csv(combine_path, index = False)
    
def find_bad_from_csv(csv_root, save_txt):
    df = pd.read_csv(csv_root)
    sess_QA = {}
    for i, item in df.iterrows():
        if item['sess'] not in sess_QA.keys():
            sess_QA[item['sess']] = []
        sess_QA[item['sess']].append(item['auto'])
    
    f = open(save_txt, 'w')
    for key in sess_QA.keys():
        if 'good' not in sess_QA[key]:
            f.write(key + '\n')
    f.close()
           
            
        
    
    