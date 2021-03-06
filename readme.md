# QA Processing tool
This repository includes the QA processing for medical image sessions. And there are some experiences and suggestions to deal with the files downloaded from XNAT. The function modules are located in *QA_tool.py*, and the examples in *test_example.py*. The test data can be found in share folder. 

Please open a issue, or contact Riqiang (riqiang.gao@vanderbilt.edu) if you have any issues or suggestions. 

## 1. Steps for QA

*For the MASI labmates who do the QA for data from XNAT. Before the QA processing, assume you already downloaded the DICOM session from XNAT or other resources. For XNAT downloading, please visit https://xnat.vanderbilt.edu/index.php/Download_data_from_XNAT to learn how to use the command line. *

The following steps code can be found at *test_example.py*. 

**Step 1**: Check the instance number of DICOMs (need to read DICOM header) if the instance number can match the number of DICOMs for the session. 

![Instance Number Failing case](./example_image/InstanceCheck.png)
<p align="center"> Fig. 1 Instance Number Failing case (click this image can see details)  </p>


**Step 2**: Check the slice distance if DICOMs (need to read DICOM header) to avoid those sessions lose slices. 

![Slice Distance Failing case](./example_image/SliceDistance.png)
<p align="center"> Fig. 2 Slice Distance Failing case (click this image can see details)  </p>



**Step 3**: filter some sessions with very limited slices (e.g. 1 or 2) but still can pass the instance number check. 

![Few Slices case](./example_image/FewSlices.png)
<p align="center"> Fig. 3 Few Slices case (click this image can see details)  </p>

**Step 4**: Find out scans unreasonablely extend Region of Interest. 

![Out of ROI case](./example_image/PhysicalLength.png)
<p align="center"> Fig. 4 Out of ROI case (click this image can see details)  </p>


**Step 5**: Use the dcm2niix tool to convert DICOMs to NIFTI by:


> dcm2niix -m n -z y -o *output_folder* *DICOM_folder*

if set the -m as n doesn't work, set the -m as y. However, here should be *very careful* to check if generated NIFTI is what you want. 


A example (2290718171-20100301) that the image is good, but dcm2niix -m n cannot successfully convert. Should use -m y.

**Step 6**: use the slicedir tool to visualize a batch of NIFTI files to double check, or use MIPAV to check one by one (time consuming). 

**Step 7**: NIFTI Orientation Check and Resolution filtering.

![Orientation Check](./example_image/orientationCheck.png)
<p align="center"> Fig. 5 Orientation Check failing case (click this image can see details)  </p>

**Step 8**: use the slicedir tool to visualize a batch of NIFTI files to double check. Or use MIPAV / ITKSNAP to check one by one (not recommended, time consuming). 


> slicesdir *NIFTI1_path* *NIFTI2_path* …… *NIFTIn_path*

For MASI lab member, you can find the binary slicesdir at /usr/share/fsl/5.0/bin/slicesdir. 

![Orientation Check](./example_image/slicesdirCheck.png)
<p align="center"> Fig. 5 Slicesdir double Check failing case (click this image can see details)  </p>

**FYI**: 


(1) when we check the slice distance in step 2, there is a parameter <slice distance difference> we should define. We can not define the difference to 0, because I find there are good images with slice distance difference. My suggestion is define the tolerance slice distance difference smaller than slice distance but larger than 0. 

(2) Maybe not all the above steps are necessary to judge a session is good or not. For example, based on my experience, I find all the sessions can pass the slice distance check (described in step 1) can always pass the instance number check (described in step 2). However, each function can provide different information for DICOM or NIFTI, and the instance number check would not take much time. 

(3) In my QA experinece, a session is good when one scan is with good quality of this session. 

 

    

