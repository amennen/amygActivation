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
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud/')

import rtCommon.utils as utils
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.structDict import StructDict
from rtCommon.dicomNiftiHandler import getTransform
from projects.amygActivation.initialize import initialize

# obtain the full path for the configuration toml file
defaultConfig = os.path.join(currPath, 'conf/amygActivation.toml')

def finalize(cfg, args):

	return

def main(argv=None):
    """
    This is the main function that is called when you run 'finialize.py'.
    
    Here, you will load the configuration settings specified in the toml configuration 
    file, initiate the class fileInterface, and set up some directories and other 
    important things through 'finalize()'
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

    # copy subject folders from server to local
    # subject-specific folder
    # everything in temp/convertedNiftis
    if args.filesremote:

        # open up the communication pipe using 'projectInterface'
        projectComm = projUtils.initProjectComm(args.commpipe, args.filesremote)

        # initiate the 'fileInterface' class, which will allow you to read and write 
        #   files and many other things using functions found in 'fileClient.py'
        #   INPUT:
        #       [1] args.filesremote (to retrieve dicom files from the remote server)
        #       [2] projectComm (communication pipe that is set up above)
        fileInterface = FileInterface(filesremote=args.filesremote, commPipes=projectComm)

        # we don't need the tmp/convertedNiftis so first remove those
        tempNiftiDir = os.path.join(cfg.server.dataDir, 'tmp/convertedNiftis/')
        projUtils.deleteFolder(tempNiftiDir)

        # next, get the final patterns data
        # maybe have them inuput for each run as an argument? specify all the run numbers or it'll look for all 
        # run folders
                run_filename = getRunFilename(cfg.sessionId, run)
        full_run_filename_to_save = os.path.join(runFolder, run_filename)
        srcPattern = cfg.
        downloadFilesFromCloud(fileInterface, srcFilePattern, outputDir, deleteAfter=False)



        #projUtils.uploadFolderToCloud(fileInterface,cfg.local.wf_dir,cfg.server.wf_dir)

        # upload ROI folder to cloud server
        #projUtils.uploadFolderToCloud(fileInterface,cfg.local.maskDir,cfg.server.maskDir)
    return 0

if __name__ == "__main__":
    """
    If 'finalize.py' is invoked as a program, then actually go through all of the 
    portions of this script. This statement is not satisfied if functions are called 
    from another script using "from finalize.py import FUNCTION"
    """    
    main()
    sys.exit(0)