# create masks

import os
import glob
import numpy as np
import json	
from datetime import datetime
from dateutil import parser
from subprocess import call
import time
import nilearn
from nilearn.masking import apply_mask
from scipy import stats
import scipy.io as sio
import pickle
import nibabel as nib
import argparse
import sys
import logging
import shutil

currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
import rtCommon.utils as utils
from rtCommon.readDicom import readDicomFromBuffer, readRetryDicomFromFileInterface
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
from initialize import initialize
from amygActivation import *
logLevel = logging.INFO

defaultConfig = os.path.join(currPath, 'conf/amygActivation.toml')


def registerMNIToNewNifti(cfg, mask, reference_native):
    # this will take given masks in MNI space and register to that days ex func scan

    # (3) combine everything with ANTs call
    # input: ROI
    # reference: today's first functional file
    # transform MNI to T1
    # transform T1 to BOLD
    # transform BOLD to BOLD
    for m in np.arange(len(cfg.MASK)):
        # rerun for each mask
        full_ROI_path = os.path.join(cfg.local.maskDir, cfg.MASK[m])
        base_ROI_name = cfg.MASK[m].split('.')[0]
        base_ROI_name_native = '{0}_space-native.nii.gz'.format(base_ROI_name)
        output_nifti_name = os.path.join(cfg.local.maskDir, base_ROI_name_native)
        command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation NearestNeighbor -d 3 -e 3 --input {0} --reference-image {1} --output {2}/{3}_space-native.nii.gz  --transform {7} --transform {6} --transform {4}/ref_2_{5}.txt -v 1'.format(full_ROI_path, reference_native, cfg.subject_reg_dir, base_ROI_name, cfg.subject_reg_dir, base_nifti_name, cfg.T1_to_BOLD, cfg.MNI_to_T1)
        #print('(3) ' + command)
        A = time.time()
        call(command, shell=True)
        B = time.time()
        print(B-A)

    return output_nifti_name 

# GO FROM MNI --> NATIVE SPACE
# reference image - example image in native space
cfg = utils.loadConfigFile(defaultConfig)
cfg.local.codeDir = os.path.join(cfg.local.rtcloudDir, 'projects', cfg.projectName)
cfg.local.dataDir = os.path.join(cfg.local.codeDir, 'data') 
cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
cfg.local.wf_dir = os.path.join(cfg.local.dataDir, cfg.bids_id, 'ses-01', 'registration')
cfg.local.maskDir = os.path.join(cfg.local.codeDir, 'ROI')
reference_native = os.path.join(cfg.local.wf_dir,'ref_image.nii.gz')
command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation NearestNeighbor -d 3 -e 3 --input {0} --reference-image {1} --output {2}/{3}_space-native.nii.gz  --transform {7} --transform {6} --transform {4}/ref_2_{5}.txt -v 1'.format(original_mask, reference_native, cfg.local.mask_dir, base_ROI_name, cfg.subject_reg_dir, base_nifti_name, cfg.T1_to_BOLD, cfg.MNI_to_T1)
A = time.time()
call(command, shell=True)
B = time.time()
print(B-A)