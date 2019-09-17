# PURPOSE: get registration transferred and ready on intel linux and cloud VM, no matter which computer you're running this on

import os
import glob
import numpy as np
from subprocess import call
import time
import nilearn
from scipy import stats
import scipy.io as sio
import pickle
import nibabel as nib
import argparse
import random
import sys
from datetime import datetime
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud/')
#sys.path.append('/Data1/code/rt-cloud/')
from rtCommon.utils import loadConfigFile
from rtCommon.structDict import StructDict
#from rtfMRI.FileInterface import FileInterface this won't work because don't have inotify--i think this caused numpy problem


# subject path from previous day
def copyClusterFileToIntel(fileOnCluster,pathOnLinux):
	"""This copies a file from the cluster to intel, assuming you're on the intel linux calling the function"""
	command = 'scp amennen@scotty:{0} {1} '.format(fileOnCluster,pathOnLinux)
	call(command,shell=True)
	#return command


def copyIntelFileToCloud(fileOnIntel,pathOnCloud,serverAddr):
	"""This copies a file from the intel computer to the cloud VM, assuming that you're on the intel linux calling the function"""
	command = 'scp -i ~/.ssh/azure_id_rsa {0} amennen@{1}:{2} '.format(fileOnIntel,serverAddr,pathOnCloud)
	call(command,shell=True)

def copyIntelFolderToCloud(folderOnIntel,pathOnCloud,serverAddr):
        """This copies a file from the intel computer to the cloud VM, assuming that you're on the intel linux calling the function"""
        command = 'scp -i ~/.ssh/azure_id_rsa -r {0} amennen@{1}:{2} '.format(folderOnIntel,serverAddr,pathOnCloud)
        call(command,shell=True)

def copyClusterFileToCluster(fileOnCluster,pathOnCluster):
    """This copies a file from the cluster to cluster, assuming you're on the cluster calling the function"""
    command = 'cp {0} {1} '.format(fileOnCluster,pathOnCluster)
    call(command,shell=True)
    #return command


def buildSubjectFoldersCluster(cfg):
    cfg.subject_full_day_path = '{0}/data/{1}/{2}'.format(cfg.cluster.codeDir,cfg.bids_id,cfg.ses_id)
    cfg.subject_offline_registration_path = '{0}/data/{1}/ses-{2:02d}/registration/'.format(cfg.cluster.codeDir,cfg.bids_id,1)
    if not os.path.exists(cfg.subject_full_day_path):
        os.makedirs(cfg.subject_full_day_path)
    if not os.path.exists(cfg.subject_offline_registration_path):
        os.makedirs(cfg.subject_offline_registration_path)
    if cfg.subjectDay == 2:
        #cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
        cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
        #if not os.path.exists(cfg.temp_nifti_dir):
        #    os.makedirs(cfg.temp_nifti_dir)
        if not os.path.exists(cfg.subject_reg_dir):
            os.makedirs(cfg.subject_reg_dir)
    return cfg


def buildSubjectFoldersIntelrt(cfg):
    cfg.subject_full_day_path = '{0}/data/{1}/{2}'.format(cfg.intelrt.codeDir,cfg.bids_id,cfg.ses_id)
    cfg.subject_offline_registration_path = '{0}/data/{1}/ses-{2:02d}/registration/'.format(cfg.intelrt.codeDir,cfg.bids_id,1)
    if not os.path.exists(cfg.subject_full_day_path):
        os.makedirs(cfg.subject_full_day_path)
    if not os.path.exists(cfg.subject_offline_registration_path):
        os.makedirs(cfg.subject_offline_registration_path)
    if cfg.subjectDay == 2:
        #cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
        cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
        #if not os.path.exists(cfg.temp_nifti_dir):
        #    os.makedirs(cfg.temp_nifti_dir)
        if not os.path.exists(cfg.subject_reg_dir):
            os.makedirs(cfg.subject_reg_dir)
    return cfg

def buildSubjectFoldersCloud(cfg):
    cfg.subject_full_day_path = '{0}/data/{1}/{2}'.format(cfg.cloud.codeDir,cfg.bids_id,cfg.ses_id)
    cfg.subject_offline_registration_path = '{0}/data/{1}/ses-{2:02d}/registration/'.format(cfg.cloud.codeDir,cfg.bids_id,1)
    if not os.path.exists(cfg.subject_full_day_path):
        os.makedirs(cfg.subject_full_day_path)
    if not os.path.exists(cfg.subject_offline_registration_path):
        os.makedirs(cfg.subject_offline_registration_path)
    if cfg.subjectDay == 2:
        # converted nifti dir is now just in data/tmp/converted_niftis
        #cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
        cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
        #if not os.path.exists(cfg.temp_nifti_dir):
        #    os.makedirs(cfg.temp_nifti_dir)
        if not os.path.exists(cfg.subject_reg_dir):
            os.makedirs(cfg.subject_reg_dir)
        cfg.intelrt.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.intelrt.codeDir,cfg.bids_id,1)
        cfg.intelrt.BOLD_to_T1= cfg.intelrt.wf_dir + 'affine.txt'
        cfg.intelrt.T1_to_MNI= cfg.intelrt.wf_dir + 'ants_t1_to_mniComposite.h5'
        cfg.intelrt.ref_BOLD=cfg.intelrt.wf_dir + 'ref_image.nii.gz'
        cfg.intelrt.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.intelrt.codeDir,cfg.bids_id,cfg.ses_id)
        cfg.intelrt.interpretationFile = '{0}/{1}_{2}_interpretation.txt'.format(cfg.intelrt.subject_full_day_path,cfg.bids_id,cfg.ses_id)
    return cfg



def main():
    random.seed(datetime.now())
    # MAKES STRUCT WITH ALL PARAMETERS IN IT
    defaultConfig = os.path.join(currPath , 'conf/faceMatching_organized.toml')
    #defaultConfig = 'conf/greenEyes_organized.toml'
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig,type=str,
                   help='experiment config file (.json or .toml)')
    argParser.add_argument('--addr', '-a', default='localhost', type=str, 
                   help='server ip address')
    args = argParser.parse_args()
    params = StructDict({'config': args.config})

    cfg = loadConfigFile(params.config)
    #cfg = loadConfigFile(defaultConfig)
    # TESTING
    cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
    # get subj
    if cfg.machine == 'intel':
        # get intel computer ready
        cfg = buildSubjectFoldersIntelrt(cfg)
        if cfg.subjectDay == 2:
            cluster_wf_dir = '{0}/derivatives/work/fmriprep_wf/single_subject_{1:03d}_wf'.format(cfg.cluster.clusterBidsDir,cfg.subjectNum)
            cluster_BOLD_to_T1 = cluster_wf_dir + '/func_preproc_ses_01_task_examplefunc_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
            cluster_T1_to_MNI = cluster_wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'
            cluster_ref_BOLD = cluster_wf_dir + '/func_preproc_ses_01_task_examplefunc_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz'
            copyClusterFileToIntel(cluster_BOLD_to_T1,cfg.subject_offline_registration_path)
            copyClusterFileToIntel(cluster_T1_to_MNI,cfg.subject_offline_registration_path)
            copyClusterFileToIntel(cluster_ref_BOLD,cfg.subject_offline_registration_path)
            # now see if you need to randomly draw the intepretation
            makeSubjectInterpretation(cfg)
            if cfg.mode == 'cloud': # also copy files to the cloud computer -- easier here to just copy entire folder
                cfg.subject_full_path = '{0}/data/{1}'.format(cfg.intelrt.codeDir,cfg.bids_id)
                locationToSend = '{0}/data/'.format(cfg.cloud.codeDir)
                if args.addr is not 'localhost':
                    copyIntelFolderToCloud(cfg.subject_full_path,locationToSend,args.addr)
                else:
                    logging.warning('YOU NEED TO INPUT CLOUD IP ADDR!!')
                    print('YOU NEED TO INPUT CLOUD IP ADDR!!')
    # elif cfg.machine == 'cloud':
    #     # get cloud computer ready
    #     cfg = buildSubjectFoldersCloud(cfg)
    #     fileInterface = FileInterface()
    #     retrieveIntelFileAndSaveToCloud(cfg.intelrt.BOLD_to_T1,cfg.subject_offline_registration_path,fileInterface)
    #     retrieveIntelFileAndSaveToCloud(cfg.intelrt.T1_to_MNI,cfg.subject_offline_registration_path,fileInterface)
    #     retrieveIntelFileAndSaveToCloud(cfg.intelrt.ref_BOLD,cfg.subject_offline_registration_path,fileInterface)
    #     retrieveInfelFileAndSaveToCloud(cfg.intelrt.interpretationFile,cfg.subject_full_day_path,fileInterface)
    elif cfg.machine == 'cluster': # running on cluster computer
        cluster_wf_dir='{0}/derivatives/work/fmriprep_wf/single_subject_{1:03d}_wf'.format(cfg.cluster.clusterBidsDir,cfg.subjectNum)
        cluster_BOLD_to_T1= cluster_wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
        cluster_T1_to_MNI= cluster_wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'
        cluster_ref_BOLD=glob.glob(cluster_wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')[0]
        cfg = buildSubjectFoldersCluster(cfg)
        copyClusterFileToCluster(cluster_BOLD_to_T1,cfg.subject_offline_registration_path)
        copyClusterFileToCluster(cluster_T1_to_MNI,cfg.subject_offline_registration_path)
        copyClusterFileToCluster(cluster_ref_BOLD,cfg.subject_offline_registration_path)
if __name__ == "__main__":
    # execute only if run as a script
    main()
