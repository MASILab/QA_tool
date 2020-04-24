# QA Processing tool
This repository includes the QA processing for medical image sessions. And there are some experiences and suggestions to deal with the files downloaded from XNAT. The function modules are located in *QA_tool.py*, and the examples in *test_example.py*. The test data can be found in share folder. 

Please contact Riqiang (riqiang.gao@vanderbilt.edu) if you have any issues or suggestions. 

## 1. Steps for QA

Before the QA processing, assume you already downloaded the DICOM session from XNAT or other resources. For XNAT downloading, please visit https://xnat.vanderbilt.edu/index.php/Download_data_from_XNAT to learn how to use the command line. 

**Step 1**: Check the instance number of DICOMs (need to read DICOM header) if the instance number can match the number of DICOMs for the session. 

**Step 2**: Check the slice distance if DICOMs (need to read DICOM header) to avoid those sessions lose slices. 

**Step 3**: filter some sessions with very limited slices (e.g. 1 or 2) but still can pass the instance number check. 

**Step 4** Use the dcm2niix tool to convert DICOMs to NIFTI by:


> dcm2niix -m n -z y -o *output_folder* *DICOM_folder*

if set the -m as n doesn't work, set the -m as y. However, here should be *very careful* to check if generated NIFTI is what you want. 

**Step 5**: use the slicedir tool to visualize a batch of NIFTI files to double check, or use MIPAV to check one by one (time consuming). 

> slicesdir *NIFTI1_path* *NIFTI2_path* …… *NIFTIn_path*

You can find the binary slicesdir at /usr/share/fsl/5.0/bin/slicesdir. 

**FYI**: 

(1) when we check the slice distance in step 2, there is a parameter <slice distance difference> we should define. We can not define the difference to 0, because I find there are good images with slice distance difference. My suggestion is define the tolerance slice distance difference smaller than slice distance but larger than 0. 

(2) Maybe not all the above steps are necessary to judge a session is good or not. For example, based on my experience, I find all the sessions can pass the slice distance check (described in step 1) can always pass the instance number check (described in step 2). However, each function can provide different information for DICOM or NIFTI, and the instance number check would not take much time. 

## 2. Suggestions and Experiences dealing with the file from XNAT

1. Some file names contain  invalid symbol (such as ')', ',' ) which might cause i/o error when process those images. 

   **My suggestion** is renaming those files before processing. Please see the function  *QA_tool/dcm2nii_project*  for example. 

2. In most of time, the DICOM file under a folder with the name "DICOM". However, in few case, DICOM files are under the folder with name "secondary". 

   **My suggestion** is renaming those files before processing. Please see the function  *QA_tool/dcm_fold*  for example.   

3. Some sessions from XNAT would have more that one folder that contains DICOM files. What we need is if only there is one folder that contains "legal" DICOM files. 

   **My suggestion** is pick the folder with largest number of DICOMs to process in the first round, and keep the record of the session if it has multiple DICOM folders or not. If yes, manually check other folders if the first round didn't pass the QA. Please see  the function  *QA_tool/dcm_fold*  for example.   

    

