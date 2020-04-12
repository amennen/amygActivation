# Purpose: get experiment ready

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
from dateutil import parser

# WHEN TESTING - COMMENT OUT
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
#WHEN TESTING: UNCOMMENT TO ADD PATH TO RT-CLOUD
#sys.path.append('/Users/amennen/github/rt-cloud/')

import rtCommon.utils as utils
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.structDict import StructDict
from rtCommon.dicomNiftiHandler import getTransform


# obtain the full path for the configuration toml file
defaultConfig = os.path.join(currPath, 'conf/amygActivation.toml')

def initialize(cfg, args):
    """ purpose: load information and add to config """

    if cfg.sessionId in (None, '') or cfg.useSessionTimestamp is True:
        cfg.useSessionTimestamp = True
        cfg.sessionId = utils.dateStr30(time.localtime())
    else:
        cfg.useSessionTimestamp = False
    # MERGE WITH PARAMS
    if args.runs != '' and args.scans != '':
        # use the run and scan numbers passed in as parameters
        cfg.runNum = [int(x) for x in args.runs.split(',')]
        cfg.scanNum = [int(x) for x in args.scans.split(',')]
    else: # when you're not specifying on the command line it's already in a list
        cfg.runNum = [int(x) for x in cfg.runNum]
        cfg.scanNum = [int(x) for x in cfg.scanNum]
    
    # GET DICOM DIRECTORY
    if cfg.buildImgPath:
        imgDirDate = datetime.now()
        dateStr = cfg.date.lower()
        if dateStr != 'now' and dateStr != 'today':
            try:
                imgDirDate = parser.parse(cfg.date)
            except ValueError as err:
                raise RequestError('Unable to parse date string {} {}'.format(cfg.date, err))
        datestr = imgDirDate.strftime("%Y%m%d")
        imgDirName = "{}.{}.{}".format(datestr, cfg.subjectName, cfg.subjectName)
        cfg.dicomDir = os.path.join(cfg.local.dicomDir,imgDirName)
    else:
        cfg.dicomDir = cfg.local.dicomDir # then the whole path was supplied
    ########
    cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
    
    # specify local directories
    cfg.local.codeDir = cfg.local.rtcloudDir + '/projects' + '/' + cfg.projectName
    cfg.local.dataDir = cfg.local.codeDir + '/' + 'data' 
    cfg.local.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.local.dataDir, cfg.bids_id, cfg.ses_id)
    cfg.local.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.local.subject_full_day_path)
    cfg.local.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.local.dataDir, cfg.bids_id, 1)
    cfg.local.maskDir = cfg.local.codeDir + '/' + 'ROI'
    cfg.subject_reg_dir = cfg.local.subject_reg_dir
    cfg.wf_dir = cfg.local.wf_dir

    if args.filesremote: # here we will need to specify separate paths for processing
        cfg.server.codeDir = cfg.server.rtcloudDir + '/projects' + '/' + cfg.projectName + '/'
        cfg.server.dataDir = cfg.server.codeDir + '/' + cfg.server.serverDataDir
        cfg.server.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.server.dataDir, cfg.bids_id, cfg.ses_id)
        cfg.server.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.server.subject_full_day_path)
        cfg.server.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.server.dataDir, cfg.bids_id, 1)
        cfg.server.maskDir = cfg.server.codeDir + '/' + 'ROI'
        cfg.subject_reg_dir = cfg.server.subject_reg_dir
        cfg.wf_dir = cfg.server.wf_dir
    cfg.ref_BOLD = cfg.wf_dir + '/' + 'ref_image.nii.gz'
    cfg.MNI_ref_filename = cfg.wf_dir + '/' + cfg.MNI_ref_BOLD 
    cfg.BOLD_to_T1 = cfg.wf_dir + '/' + 'affine.txt'
    cfg.T1_to_MNI = cfg.wf_dir + '/' + 'ants_t1_to_mniComposite.h5'
    # get conversion to flip dicom to nifti files
    cfg.axesTransform = getTransform(('L', 'A', 'S'),('P', 'L', 'S'))
    return cfg

# this project will already have the local files included on it
# def buildSubjectFoldersOnLocal(cfg):
#     """This function transfers registration files from the fmriprep workflow path to the experiment data path
#     on the Linux machine in the scanner suite"""
#     if not os.path.exists(cfg.local.subject_full_day_path):
#         os.makedirs(cfg.local.subject_full_day_path)
#     if not os.path.exists(cfg.local.wf_dir):
#         os.makedirs(cfg.local.wf_dir)
#         print('***************************************')
#         print('CREATING WF DIRECTORY %s' % cfg.local.wf_dir)
#     print('***************************************')
#     print('MAKE SURE YOU HAVE ALREADY TRANSFERRED FMRIPREP REGISTRATION OUTPUTS HERE TO %s' % cfg.local.wf_dir)
#     print('IF YOUR FMRIPREP WORK/WORKFLOW DIR IS wf_dir, FIND OUTPUTS IN:')
#     print('T1->MNI: wf_dir/func_preproc_ses_01_task_examplefunc_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt')
#     print('BOLD->T1: wf_dir/func_preproc_ses_01_task_examplefunc_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt')
#     print('example func: wf_dir/func_preproc_ses_01_task_examplefunc_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')
#     return 

def buildSubjectFoldersOnServer(cfg):
    """This function transfers registration files from the experiment data path on the 
     Linux machine in the scanner suite to the cloud server where data is processed in real-time"""
    if not os.path.exists(cfg.server.subject_full_day_path):
        os.makedirs(cfg.server.subject_full_day_path)
    if not os.path.exists(cfg.server.wf_dir):
        os.makedirs(cfg.server.wf_dir)
    if not os.path.exists(cfg.server.subject_reg_dir):
        os.mkdir(cfg.server.subject_reg_dir)
        print('CREATING REGISTRATION DIRECTORY %s' % cfg.server.subject_reg_dir)
    return 

####################################################################################
# defaultConfig = 'conf/amygActivation.toml'
# args = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'commpipe': None, 'filesremote': True})
####################################################################################

def main(argv=None):
    """
    This is the main function that is called when you run 'intialize.py'.
    
    Here, you will load the configuration settings specified in the toml configuration 
    file, initiate the class fileInterface, and set up some directories and other 
    important things through 'initialize()'
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    # This parameter is used for projectInterface
    argParser.add_argument('--commpipe', '-q', default=None, type=str,
                           help='Named pipe to communicate with projectInterface')
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                           help='retrieve files from the remote server')
    argParser.add_argument('--addr', '-a', default='localhost', type=str, 
               help='server ip address')
    argParser.add_argument('--runs', '-r', default='', type=str,
                       help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                       help='Comma separated list of scan number')
    args = argParser.parse_args(argv)

    # load the experiment configuration file
    cfg = utils.loadConfigFile(args.config)
    cfg = initialize(cfg, args)

    # build subject folders on server
    if args.filesremote:
        buildSubjectFoldersOnServer(cfg)

        # open up the communication pipe using 'projectInterface'
        projectComm = projUtils.initProjectComm(args.commpipe, args.filesremote)

        # initiate the 'fileInterface' class, which will allow you to read and write 
        #   files and many other things using functions found in 'fileClient.py'
        #   INPUT:
        #       [1] args.filesremote (to retrieve dicom files from the remote server)
        #       [2] projectComm (communication pipe that is set up above)
        fileInterface = FileInterface(filesremote=args.filesremote, commPipes=projectComm)

        # next, transfer transformation files from local --> server for online processing
        projUtils.uploadFolderToCloud(fileInterface,cfg.local.wf_dir,cfg.server.wf_dir)

        # upload ROI folder to cloud server
        projUtils.uploadFolderToCloud(fileInterface,cfg.local.maskDir,cfg.server.maskDir)
    return 0

if __name__ == "__main__":
    """
    If 'initalize.py' is invoked as a program, then actually go through all of the 
    portions of this script. This statement is not satisfied if functions are called 
    from another script using "from initalize.py import FUNCTION"
    """    
    main()
    sys.exit(0)
