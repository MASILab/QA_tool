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
import skimage

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
    cmd_pre = "dcm2niix -m y "                               #    "dcm2nii -r n "           # dcm2niix -y z    # dcm2niix -m y                        
    cmd = cmd_pre + "-o " + dst_root + " " + src_root
    os.system(cmd)
    print (cmd)
    
def dcm_instance(dcm_root):
    '''
    check a DICOM folder if it is valid
    '''
    dcm_list = glob(os.path.join(dcm_root, '*.dcm'))
    slicePos = []
    instanceN = []
    for i in range(len(dcm_list)):
        ds = pydicom.dcmread(dcm_list[i])
        #slicePos.append(ds.SliceLocation)
        instanceN.append(ds[0x20, 0x13].value)
    print ('max and min of instanceN', max(instanceN), min(instanceN))
    return len(instanceN), max(instanceN) - min(instanceN) + 1,  max(instanceN) - min(instanceN) + 1 - len(instanceN)

def dcm_slicedistance(dcm_root):
    dcm_list = os.listdir(dcm_root)
    dcm_list = glob(os.path.join(dcm_root, '*.dcm'))
    ds_list = []
    for i in range(len(dcm_list)):
        ds = pydicom.dcmread(dcm_list[i])
        ds_list.append(ds.SliceLocation)

    ds_sort = sorted(ds_list, reverse = True)
    res = 1
    for i in range(0, len(ds_sort) - 2):
        #print ((ds_sort[i] - ds_sort[i + 1]), (ds_sort[i + 1] - ds_sort[i + 2]))
        if not abs((ds_sort[i] - ds_sort[i + 1]) - (ds_sort[i + 1] - ds_sort[i + 2])) < (ds_sort[0] - ds_sort[1]):
            res = 0
    return res

def dcm_lenbody(dcm_root):
    dcm_list = os.listdir(dcm_root)
    ds_list = []
    for i in range(len(dcm_list)):
        ds = pydicom.dcmread(dcm_root + '/' + dcm_list[i])
        ds_list.append(ds.SliceLocation)

    ds_sort = sorted(ds_list, reverse = True)

    return ds_sort[0] - ds_sort[-1]

def lenbody_fold(fold_root, save_csv_path):
    subj_list = os.listdir(fold_root)
    sess, single_folder, length = [], [], []
    #subj = []
    for i in range(0, len(subj_list)):
       # if i > 3: break
        if i % 10 == 0: print (i, len(subj_list))
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            #print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            if len(instance_list) == 1:
                single_folder.append(1)
            else:
                single_folder.append(0)
            size_list = []
            try:
                for k in range(len(instance_list)):
                    #print (sess_path + '/' + instance_list[k])

                    size = len(os.listdir(sess_path + '/' + instance_list[k] + '/DICOM'))
                    #print (size)
                    size_list.append(size)
                max_index = size_list.index(max(size_list))
                l = dcm_lenbody(sess_path + '/' + instance_list[max_index] + '/DICOM')
                length.append(l)
            except:
                length.append('')
                print ("length error", subj_list[i], sess_list[j])
    data = pd.DataFrame()
    data['sess'] = sess
    data['single_folder'] = single_folder
    data['length'] = length
    data.to_csv(save_csv_path, index = False)

def lenbody_fold_NLST(fold_root, save_csv_path):
    subj_list = os.listdir(fold_root)
    subj, sess, length = [], [], []
    for i in range(len(subj_list)):
        #if i > 5: break
        if i % 10 == 0: print (i, len(subj_list))
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            subj.append(subj_list[i])
            #print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            size_list = []
            try:
                for k in range(len(instance_list)):
                    #print (sess_path + '/' + instance_list[k])

                    size = len(os.listdir(sess_path + '/' + instance_list[k]))
                    #print (size)
                    size_list.append(size)
                max_index = size_list.index(max(size_list))
                l = dcm_lenbody(sess_path + '/' + instance_list[max_index])
                length.append(l)
            except:
                length.append('')
                print ("length error", subj_list[i], sess_list[j])
    data = pd.DataFrame()
    data['subj'] = subj
    data['sess'] = sess
    data['length'] = length
    data.to_csv(save_csv_path, index = False)

def check_affine(nii_path):
    img_nib = nib.load(nii_path)
    aff_mat = img_nib.affine
    if aff_mat[0][0] < 0 and aff_mat[1][1] > 0 and aff_mat[2][2] > 0 and aff_mat[3][3] > 0:
        return 1
    else:
        return 0
    ## please use fslreorient2std to make the orientation correct. 
    
def resolution_max(nii_path):
    img_nib = nib.load(nii_path)
    aff_mat = img_nib.affine
    resolution = [abs(aff_mat[0][0]), abs(aff_mat[1][1]), abs(aff_mat[2][2])]
    return max(resolution)
    
def check_affine_fold(fold_root, save_csv_path): 
    # different from sliceDis_fold and dcm_fold, this is for nifti folder.
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
    
def check_affine_fold_NLST(fold_root, save_csv_path): 
    # different from sliceDis_fold and dcm_fold, this is for nifti folder.
    subj_list = os.listdir(fold_root)
    res_list,SESS_list = [], []
    for i in range(len(subj_list)):
        #if i > 10: break
        if i % 10 == 0: print (i, len(subj_list))
        sess_list = os.listdir(fold_root + '/' + subj_list[i])
        for j in range(len(sess_list)):
            SESS_list.append(subj_list[i] + 'time' + sess_list[j])
            nib_paths = glob(fold_root + '/' + subj_list[i] + '/' + sess_list[j] + '/*.nii.gz')
            if len(nib_paths) < 1:
                res_list.append('')
            else:
                nib_path = nib_paths[0]
                res = check_affine(nib_path)
                res_list.append(res)        
    data = pd.DataFrame()
    data['id'] = SESS_list
    data['affine_check'] = res_list
    data.to_csv(save_csv_path, index = False)
    
def max_res_fold_NLST(fold_root, save_csv_path): 
    # different from sliceDis_fold and dcm_fold, this is for nifti folder.
    subj_list = os.listdir(fold_root)
    res_list,SESS_list = [], []
    for i in range(len(subj_list)):
        #if i > 10: break
        if i % 10 == 0: print (i, len(subj_list))
        sess_list = os.listdir(fold_root + '/' + subj_list[i])
        for j in range(len(sess_list)):
            SESS_list.append(subj_list[i] + 'time' + sess_list[j])
            nib_paths = glob(fold_root + '/' + subj_list[i] + '/' + sess_list[j] + '/*.nii.gz')
            if len(nib_paths) < 1:
                res_list.append('')
            else:
                nib_path = nib_paths[0]
                res = resolution_max(nib_path)
                res_list.append(res)        
    data = pd.DataFrame()
    data['id'] = SESS_list
    data['res_max'] = res_list
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
                try:
                    same = dcm_slicedistance(sess_path + '/file0/DICOM')
                    diff.append(same)
                except:
                    diff.append('')
                    
                    print ('dicom error')
    data = pd.DataFrame()
    data['sess'] = sess
    data['single_folder'] = single_folder
    data['distance_check'] = diff
    data.to_csv(save_csv_path, index = False)
    
def sliceDis_fold_NLST(fold_root, save_csv_path):
    subj_list = os.listdir(fold_root)
    sess, single_folder, diff = [], [], []
    for i in range(0, len(subj_list)):
        if i % 10 == 0: print (i, len(subj_list))
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            #print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            if len(instance_list) == 1:
                single_folder.append(1)
            else:
                single_folder.append(0)
            size_list = []
            
            
            try:
                for k in range(len(instance_list)):
                    #print (sess_path + '/' + instance_list[k])

                    size = len(os.listdir(sess_path + '/' + instance_list[k]))
                    #print (size)
                    size_list.append(size)
                max_index = size_list.index(max(size_list))
                same = dcm_slicedistance(sess_path + '/' + instance_list[max_index])
                diff.append(same)
            except:
                diff.append('')
                print (sess_list[j], 'dicom error')
    data = pd.DataFrame()
    data['sess'] = sess
    data['single_folder'] = single_folder
    data['distance_check'] = diff
    data.to_csv(save_csv_path, index = False)
    
def instanceN_fold(fold_root, save_csv_path): # instanceN_fold
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
    
def instanceN_fold_NLST(fold_root, save_csv_path): # instanceN_fold
    subj_list = os.listdir(fold_root)
    sess, single_folder, instanceN, dicomN, diff = [], [], [], [],[]
    for i in range(0, len(subj_list)):
        if i % 10 == 0: print (i, len(subj_list))
        subj_path = fold_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess.append(sess_list[j])
            #print ('(i, j): ',i, j, sess_list[j])
            sess_path = subj_path + '/' + sess_list[j]
            instance_list = os.listdir(sess_path)
            if len(instance_list) == 1:
                single_folder.append(1)
            else:
                single_folder.append(0)
            size_list = []
            
            try:
                for k in range(len(instance_list)):

                    size = len(os.listdir(sess_path + '/' + instance_list[k]))

                    size_list.append(size)
                max_index = size_list.index(max(size_list))
                inst_n, dicom_n, same = dcm_instance(sess_path + '/' + instance_list[max_index])
                instanceN.append(inst_n)
                dicomN.append(dicom_n)
                diff.append(same)
            except:
                instanceN.append('')
                dicomN.append('')
                diff.append('')
                print (sess_list[j], 'dicom error')
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
                        

                        
def cp_to_combine(data_root, new_root, QA_csv):  # this copy the QA data to combine. 
    df = pd.read_csv(QA_csv)
    for i, item in df.iterrows():
        if item['usability'] != item['usability']:
            ori_path = data_root + '/' + str(item['subject']) + '/' + str(item['time']) + '/' + str(item['niftiName']) 
            
            new_path = new_root + '/' + str(item['subject']) + 'time' + str(item['time'])
            mkdir(new_path)
            os.system('mv ' + ori_path + ' ' + new_path)
            print ('mv ' + ori_path + ' ' + new_path)



            
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
    
def combine_disQA_csv(QA_paths, combine_path):
    sess_list, single_folder_list, distance_check_list = [], [], []
    for path in QA_paths:
        df = pd.read_csv(path)
        sess_list += df['sess'].tolist()
        single_folder_list += df['single_folder'].tolist()
        distance_check_list += df['distance_check'].tolist()
    data = pd.DataFrame()
    data['sess'] = sess_list
    data['single_folder'] = single_folder_list
    data['distance_check'] = distance_check_list
    
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
           
def find_bad_from_discsv(csv_root, save_txt):
    df = pd.read_csv(csv_root)
    sess_QA = {}
    for i, item in df.iterrows():
        sess_vec = re.split('[-_]', item['sess']) # if spore [-], if mcl [_-]
        sess = sess_vec[0] + '-' + sess_vec[-1]
        if sess not in sess_QA.keys():
            sess_QA[sess] = []
        
        sess_QA[sess].append(item['distance_check'])
    
    f = open(save_txt, 'w')
    for key in sess_QA.keys():
        if 1 not in sess_QA[key]:
            f.write(key + '\n')
    f.close()  
    
def QA_npy(npy_path, save_path, img_size):
    img = np.load(npy_path)[0]

    img_shape = img.shape
    new_img = np.zeros((3 * img_size, 3 * img_size))

    tmp_dim = img_shape[0]
    for i in range(3):
        tmp_ind = int(tmp_dim / 4 * (i +1))
        tmp_img = transform.resize(img[tmp_ind, :, :], (img_size, img_size))
        new_img[: img_size, i * img_size : (i+1) *img_size] = tmp_img

    tmp_dim = img_shape[1]
    for i in range(3):
        tmp_ind = int(tmp_dim / 4 * (i +1))
        tmp_img = transform.resize(img[:, tmp_ind,  :], (img_size, img_size))
        new_img[img_size : 2 * img_size, i * img_size : (i+1) *img_size] = tmp_img

    tmp_dim = img_shape[2]
    for i in range(3):
        tmp_ind = int(tmp_dim / 4 * (i +1))
        tmp_img = transform.resize(img[:,  :, tmp_ind], (img_size, img_size))
        new_img[2 * img_size : 3 * img_size, i * img_size : (i+1) *img_size] = tmp_img
    skimage.io.imsave(save_path, new_img)
    #plt.imshow(new_img)
    
def QA_npy_fold(npy_root, save_root, img_size):
    npy_list = os.listdir(npy_root)
    
    npy_list = [i for i in npy_list if 'clean' in i]
    
    for i in range(len(npy_list)):
        #if i > 3: break
        npy_path = npy_list[i]
        print (i, len(npy_list), npy_list[i])
        save_path = save_root + '/' + npy_path.replace('.npy', '.png')
        QA_npy(npy_root + '/' + npy_path, save_path, img_size)
        
        

    


    
