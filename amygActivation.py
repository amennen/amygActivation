# main script to run the processing of the experiment

import os
import glob
import numpy as np
import json	
import datetime
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


#WHEN TESTING
sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
# UNCOMMENT WHEN NO LONGER TESTING
# currPath = os.path.dirname(os.path.realpath(__file__))
# rootPath = os.path.dirname(os.path.dirname(currPath))
# sys.path.append(rootPath)
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile
from rtCommon.readDicom import readDicomFromBuffer, readRetryDicomFromFileInterface
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
# in tests directory can see test script

logLevel = logging.INFO

def initializeAmygActivation(configFile, args):
    # load subject information
    # create directories for new niftis
    # purpose: load information and add to configuration things that you won't want to do each time a new file comes in
    # TO RUN AT THE START OF EACH RUN

    cfg = loadConfigFile(configFile)
    if cfg.sessionId in (None, '') or cfg.useSessionTimestamp is True:
        cfg.useSessionTimestamp = True
        cfg.sessionId = dateStr30(time.localtime())
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
        imgDirDate = datetime.datetime.now()
        dateStr = cfg.date.lower()
        if dateStr != 'now' and dateStr != 'today':
            try:
                imgDirDate = parser.parse(cfg.date)
            except ValueError as err:
                raise RequestError('Unable to parse date string {} {}'.format(cfg.date, err))
        datestr = imgDirDate.strftime("%Y%m%d")
        imgDirName = "{}.{}.{}".format(datestr, cfg.subjectName, cfg.subjectName)
        if cfg.mode != 'debug':
            cfg.dicomDir = os.path.join(cfg.intelrt.imgDir, imgDirName)
            cfg.dicomNamePattern = cfg.intelrt.dicomNamePattern
        else:
            cfg.dicomDir = os.path.join(cfg.cluster.imgDir, imgDirName)
            cfg.dicomNamePattern = cfg.cluster.dicomNamePattern
    else: # if you're naming the full folder directly
        if cfg.mode != 'debug':
            cfg.dicomDir = cfg.intelrt.imgDir # then the whole path was supplied
            cfg.dicomNamePattern = cfg.intelrt.dicomNamePattern
        else:
            cfg.dicomDir = cfg.cluster.imgDir
            cfg.dicomNamePattern = cfg.cluster.dicomNamePattern

	########
    cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
    if cfg.mode == 'local':
        # then all processing is happening on linux too
        cfg.dataDir = cfg.intelrt.codeDir + 'data'
        cfg.mask_filename = os.path.join(cfg.intelrt.maskDir, cfg.MASK)
        cfg.MNI_ref_filename = os.path.join(cfg.intelrt.maskDir, cfg.MNI_ref_BOLD)
    elif cfg.mode == 'cloud':
        cfg.dataDir = cfg.cloud.codeDir + 'data'
        cfg.mask_filename = os.path.join(cfg.cloud.maskDir, cfg.MASK)
        cfg.MNI_ref_filename = os.path.join(cfg.cloud.maskDir, cfg.MNI_ref_BOLD)
        cfg.intelrt.subject_full_day_path = '{0}/data/{1}/{2}'.format(cfg.intelrt.codeDir, cfg.bids_id, cfg.ses_id)
    elif cfg.mode == 'debug':
        cfg.dataDir = cfg.cluster.codeDir + '/data'
        cfg.mask_filename = os.path.join(cfg.cluster.maskDir, cfg.MASK)
        cfg.MNI_ref_filename = os.path.join(cfg.cluster.maskDir, cfg.MNI_ref_BOLD)

	
    cfg.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.dataDir, cfg.bids_id, cfg.ses_id)
    cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
    # check that this directory exists
    if not os.path.exists(cfg.subject_reg_dir):
        os.mkdir(cfg.subject_reg_dir)
        print('CREATING REGISTRATION DIRECTORY %s' % cfg.subject_reg_dir)
	# REGISTRATION THINGS
    cfg.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.dataDir, cfg.bids_id, 1)
    cfg.BOLD_to_T1= cfg.wf_dir + 'affine.txt'
    cfg.T1_to_MNI= cfg.wf_dir + 'ants_t1_to_mniComposite.h5'
    cfg.ref_BOLD=cfg.wf_dir + 'ref_image.nii.gz'

    # GET CONVERSION FOR HOW TO FLIP MATRICES
    cfg.axesTransform = getTransform()
    ###### BUILD SUBJECT FOLDERS #######
    return cfg

def getRegressorName(runNum):
    """"Return station classification filename"""
    # this is the actual run number, 1-based
    filename = "regressor_run-{0:02d}.mat".format(runNum)
    return filename

def makeRunReg(cfg,runNum,runFolder,runFolderLinux,saveMat=1,):
    """ make regression for neurofeedback to use """
    # runIndex is 0-based, we'll save as the actual run name
    # get # TRs duration from config file
    nReps = int(cfg.nReps)
    nTR_block = int(cfg.nTR_block)
    total_n_blocks = nReps*3 + 1
    total_n_TRs = total_n_blocks * nTR_block
    regressor = np.zeros((total_n_TRs,))
    # REST = 0
    # NEUROFEEDBACK = 1
    # MATH = 2
    val_dict = {}
    val_dict['REST'] = int(cfg.REST)
    val_dict['HAPPY'] = int(cfg.HAPPY)
    val_dict['MATH'] = int(cfg.MATH)
    for r in np.arange(nReps):
        first_start = r*(nTR_block*3)
        first_end = first_start + nTR_block
        regressor[first_start:first_end] = val_dict[cfg.order_block[0]]  
        regressor[first_end:first_end+nTR_block] =  val_dict[cfg.order_block[1]]  
        regressor[first_end+nTR_block:first_end+(2*nTR_block)] =  val_dict[cfg.order_block[2]]  
    # save regressor as .mat to load with display
    if saveMat:
        filename = getRegressorName(runNum)
        full_name = "{0}/{1}".format(runFolder,filename)
        regData = StructDict()
        regData.regressor = regressor
        sio.savemat(full_name, regData, appendmat=False)
        if cfg.mode == 'cloud':
            # make sure to save on the local Linux as well if processing on the cloud
            full_name = "{0}/{1}".format(runFolderLinux,filename)
            sio.savemat(full_name,regData,appendmat=False)
    return regressor

def findConditionTR(regressor, condition):
    """ Return TRs of the given condition """
    allTRs = np.argwhere(regressor == condition)[:,0]
    return allTRs

def getTransform():
	target_orientation = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
	dicom_orientation = nib.orientations.axcodes2ornt(('P', 'L', 'S'))
	transform = nib.orientations.ornt_transform(dicom_orientation, target_orientation)
	return transform


def convertToNifti(TRnum, scanNum, cfg, dicomData):
    #anonymizedDicom = anonymizeDicom(dicomData) # should be anonymized already
    scanNumStr = str(scanNum).zfill(2)
    fileNumStr = str(TRnum).zfill(3)
    expected_dicom_name = cfg.dicomNamePattern.format(scanNumStr, fileNumStr)
    tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
    nameToSaveNifti = expected_dicom_name.split('.')[0] + '.nii.gz'
    fullNiftiFilename = os.path.join(tempNiftiDir, nameToSaveNifti)
    if not os.path.isfile(fullNiftiFilename): # only convert if haven't done so yet (check if doesn't exist)
       fullNiftiFilename = dnh.saveAsNiftiImage(dicomData, expected_dicom_name, cfg)
    else:
        print('SKIPPING CONVERSION FOR EXISTING NIFTI {}'.format(fullNiftiFilename))
    return fullNiftiFilename
    # ask about nifti conversion or not

def registerNewNiftiToMNI(cfg, full_nifti_name):
    # should operate over each TR
    # needs full path of nifti file to register
    base_nifti_name = full_nifti_name.split('/')[-1].split('.')[0]
    output_nifti_name = '{0}{1}_space-MNI.nii.gz'.format(cfg.subject_reg_dir, base_nifti_name)
    if not os.path.isfile(output_nifti_name): # only run this code if the file doesn't exist already
        # (1) run mcflirt with motion correction to align to bold reference
        command = 'mcflirt -in {0} -reffile {1} -out {2}{3}_MC -mats'.format(full_nifti_name, cfg.ref_BOLD, cfg.subject_reg_dir, base_nifti_name)
        #print('(1) ' + command)
        A = time.time()
        call(command, shell=True)
        B = time.time()
        print(B-A)

        # (2) run c3daffine tool to convert .mat to .txt
        command = 'c3d_affine_tool -ref {0} -src {1} {2}{3}_MC.mat/MAT_0000 -fsl2ras -oitk {4}{5}_2ref.txt'.format(cfg.ref_BOLD, full_nifti_name, cfg.subject_reg_dir, base_nifti_name, cfg.subject_reg_dir, base_nifti_name)
        #print('(2) ' + command)
        A = time.time()
        call(command, shell=True)
        B = time.time()
        print(B-A)

        # (3) combine everything with ANTs call
        command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input {0} --reference-image {1} --output {2}{3}_space-MNI.nii.gz --transform {4}{5}_2ref.txt --transform {6} --transform {7} -v 1'.format(full_nifti_name, cfg.MNI_ref_filename, cfg.subject_reg_dir, base_nifti_name, cfg.subject_reg_dir, base_nifti_name, cfg.BOLD_to_T1, cfg.T1_to_MNI)
        #print('(3) ' + command)
        A = time.time()
        call(command, shell=True)
        B = time.time()
        print(B-A)
    else:
        print('SKIPPING REGISTRATION FOR EXISTING NIFTI {}'.format(output_nifti_name))

    return output_nifti_name 

def getDicomFileName(cfg, scanNum, fileNum):
    if scanNum < 0:
        raise ValidationError("ScanNumber not supplied of invalid {}".format(scanNum))
    scanNumStr = str(scanNum).zfill(2)
    fileNumStr = str(fileNum).zfill(3)
    if cfg.dicomNamePattern is None:
        raise InvocationError("Missing config settings dicomNamePattern")
    fileName = cfg.dicomNamePattern.format(scanNumStr, fileNumStr)
    fullFileName = os.path.join(cfg.dicomDir, fileName)
    return fullFileName


def getOutputFilename(runId, TRindex):
	""""Return station classification filename"""
	filename = "percentChange_run{}_TRindex{}.txt".format(runId, TRindex)
	return filename

def getRunFilename(sessionId, runId):
	"""Return run filename given session and run"""
	filename = "patternsData_r{}_{}_py.mat".format(runId, sessionId)
	return filename

def retrieveLocalFileAndSaveToCloud(localFilePath, pathToSaveOnCloud, fileInterface):
	data = fileInterface.getFile(localFilePath)
	writeFile(pathToSaveOnCloud,data)

def findBadVoxels(cfg, dataMatrix, previous_badVoxels=None):
    # remove bad voxels
    # bad voxel criteria: (1) if raw signal < 100 OR std is < 1E-3 ( I think we're going to set it equal to 0 anyway)
    # remove story TRs
    # remove story average
    std = np.std(dataMatrix, axis=1, ddof=1)
    non_changing_voxels = np.argwhere(std < 1E-3)
    low_value_voxels = np.argwhere(np.min(dataMatrix, axis=1) < 100)
    badVoxels = np.unique(np.concatenate((non_changing_voxels, low_value_voxels)))
    # now combine with previously made badvoxels
    if previous_badVoxels is not None:
        updated_badVoxels = np.unique(np.concatenate((previous_badVoxels, badVoxels)))
    else:
        updated_badVoxels = badVoxels
    return updated_badVoxels



def makeRunHeader(cfg, runIndex): 
    # Output header 
    now = datetime.datetime.now() 
    print('**************************************************************************************************')
    print('* amygActivation v.1.0') 
    print('* Date/Time: ' + now.isoformat()) 
    print('* Subject Number: ' + str(cfg.subjectNum)) 
    print('* Subject Name: ' + str(cfg.subjectName)) 
    print('* Run Number: ' + str(cfg.runNum[runIndex])) 
    print('* Scan Number: ' + str(cfg.scanNum[runIndex])) 
    print('* Real-Time Data: ' + str(cfg.rtData))     
    print('* Mode: ' + str(cfg.mode)) 
    print('* Machine: ' + str(cfg.machine)) 
    print('* Dicom directory: ' + str(cfg.dicomDir)) 
    print('**************************************************************************************************')
    # prepare for TR sequence 
    print('{:10s}{:10s}{:10s}{:10s}'.format('run', 'filenum', 'TRindex', 'percent_change')) 
    return  

def makeTRHeader(cfg, runIndex, TRFilenum, TRindex, percent_change):
    print('{:<10.0f}{:<10d}{:<10d}{:<10.3f}'.format(
        cfg.runNum[runIndex], TRFilenum, TRindex, percent_change))
    return

def createRunFolder(cfg,runNum):
    runId = 'run-{0:02d}'.format(runNum)
    runFolder = cfg.subject_full_day_path + '/' + runId
    if not os.path.exists(runFolder):
        os.makedirs(runFolder)
    runFolderLinux = -1
    if cfg.mode == 'cloud':
        # if using cloud, also make this folder on the local Linux machine"
        runFolderLinux = cfg.intelrt.subject_full_day_path + '/' + runId
        if not os.path.exists(runFolderLinux):
            os.makedirs(runFolderLinux)

    return runFolder, runFolderLinux

def createTmpFolder(cfg):
    tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
    if not os.path.exists(tempNiftiDir):
        os.makedirs(tempNiftiDir)
    return

def deleteTmpFiles(cfg):
    tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
    if os.path.exists(tempNiftiDir):
        shutil.rmtree(tempNiftiDir)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('DELETING ALL NIFTIS IN tmp/convertedNiftis')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return

def getAvgSignal(TRs_to_average, runData, TRindex, cfg):
    """Average previous data for each voxel"""
    average_signal = np.mean(runData.all_data[:, TRs_to_average], axis=1)
    if TRindex ==0 or len(runData.badVoxels) == 0:
        runData.badVoxels = findBadVoxels(cfg, runData.all_data[:, 0:TRindex+1])
    else:
        runData.badVoxels = findBadVoxels(cfg, runData.all_data[:, 0:TRindex+1], runData.badVoxels)
    if len(runData.badVoxels) > 0:
        average_signal[runData.badVoxels] = np.nan
    return average_signal, runData

def calculatePercentChange(average_data, current_data):
    """ Calculate precent signal change compared to most recent fixation block"""
    percent_change = (current_data - average_data)/average_data
    avg_percent_change = np.nanmean(percent_change)*100
    if avg_percent_change < 0:
        avg_percent_change = 0
    return avg_percent_change

def split_tol(test_list, tol): 
    res = [] 
    last = test_list[0] 
    for ele in test_list: 
        if ele-last > tol: 
            yield res 
            res = [] 
        res.append(ele) 
        last = ele 
    yield res 

# testing code--debug mode -- run in amygActivation directory
from amygActivation import *
defaultConfig = 'conf/amygActivation_cluster.toml'
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'commpipe': 'None', 'filesremote': False})
cfg = initializeAmygActivation(params.config, params)
args = StructDict()
args.filesremote = False



def main():
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    logging.info('amygActivation: first log message!')
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                       help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='', type=str,
                       help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                       help='Comma separated list of scan number')
    # creates pipe communication link to send/request responses through web pipe
    argParser.add_argument('--commpipe', '-q', default=None, type=str,
                       help='Named pipe to communicate with projectInterface')
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                       help='dicom files retrieved from remote server')
    argParser.add_argument('--deleteTmpNifti', '-d', default='1', type=str,
                       help='Set to 0 if rerunning during a single scanning after error')

    args = argParser.parse_args()
    print(args)
    cfg = initializeAmygActivation(args.config, args)

    createTmpFolder(cfg)
    # DELETE ALL FILES IF FLAGGED (DEFAULT) # 
    if args.deleteTmpNifti == '1':
        deleteTmpFiles(cfg)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('NOT DELETING NIFTIS IN tmp/convertedNiftis')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # comm pipe
    projComm = projUtils.initProjectComm(args.commpipe, args.filesremote)
    fileInterface = FileInterface(filesremote=args.filesremote, commPipes=projComm)
    # intialize watching in particular directory
    fileInterface.initWatch(cfg.dicomDir, cfg.dicomNamePattern, cfg.minExpectedDicomSize) 
    # Young 2014 run struture:
    # blocks of rest, happy, count each 40 seconds
    # continuous feedback during the whole happy block
    # no feedback condition - same happy task but no feedback 
    # we're going to have even/odd judgements instead of counting down so it's not stressful
    # 13 blocks of 40 seconds each = 520 seconds (347 TRs) - put 10 TRs from first trigger and then 5 at the end
    # 362 TRs total

    # let's make them 42 seconds each for the TRs to be right --> 379 TRs total


    #### MAIN PROCESSING ###
    nRuns = len(cfg.runNum)
    for runIndex in np.arange(nRuns):
        # Steps that we have to do:
        # 1. load run regressor X - ** make run regressor that has TRs - 
        # 2. find the happy face trials (happy) X
        # 3. find the rest TRs right before each one  X
        # At every TR --> register to MNI, mask, etc
        # 4. zscore previous rest data (convert + register like before)
        # 5. calculate percent signal change over ROI
        # 6. save as a text file (Every TR-- display can smooth it)
        
        runNum = cfg.runNum[runIndex]# this will be 1-based now!! it will be the actual run number in case it's out of order
        makeRunHeader(cfg, runIndex)
        run = cfg.runNum[runIndex]
        # create run folder
        runFolder, runFolderLinux = createRunFolder(cfg,runNum)
        scanNum = cfg.scanNum[runIndex]
        regressor = makeRunReg(cfg,runNum,runFolder,runFolderLinux,saveMat=1)

        happy_TRs = findConditionTR(regressor,int(cfg.HAPPY))
        happy_TRs_shifted = happy_TRs  + cfg.nTR_shift
        happy_TRs_shifted_filenum = happy_TRs_shifted + cfg.nTR_skip # to account for first 10 files that we're skipping
        happy_blocks = list(split_tol(happy_TRs_shifted,1)) 
        TR_per_block = cfg.nTR_block

        fixation_TRs = findConditionTR(regressor,int(cfg.REST)) 
        fixation_TRs_shifted = fixation_TRs + cfg.nTR_shift
        fixation_blocks = list(split_tol(fixation_TRs_shifted,1)) 

        runData = StructDict()
        runData.all_data = np.zeros((cfg.nVox, cfg.nTR_run - cfg.nTR_skip))
        runData.percent_change = np.zeros((cfg.nTR_run - cfg.nTR_skip,))
        runData.percent_change[:] = np.nan
        runData.badVoxels = np.array([])
        

        TRindex = 0
        for TRFilenum in np.arange(cfg.nTR_skip+1, cfg.nTR_run+1): # iterate through all TRs
            if TRFilenum == cfg.nTR_skip+1: # wait until run starts
                timeout_file = 180
            else:
                timeout_file = 5
            A = time.time()
            dicomData = readRetryDicomFromFileInterface(fileInterface, getDicomFileName(cfg, scanNum, TRFilenum), timeout=timeout_file)
            full_nifti_name = convertToNifti(TRFilenum, scanNum, cfg, dicomData)
            registeredFileName = registerNewNiftiToMNI(cfg, full_nifti_name)
            maskedData = apply_mask(registeredFileName, cfg.mask_filename)
            runData.all_data[:, TRindex] = maskedData
            B = time.time()
            print('read to mask time: {:5f}'.format(B-A))

            if TRindex in happy_TRs_shifted: # we're at a happy block
                # now take previous fixation block for z scoring 
                this_block = [b for b in np.arange(4) if TRindex in happy_blocks[b]][0]
                fixation_this_block = fixation_blocks[this_block]
                avg_activity, runData = getAvgSignal(fixation_this_block, runData, TRindex, cfg)
                runData.percent_change[TRindex] = calculatePercentChange(avg_activity, runData.all_data[:, TRindex])
                
                text_to_save = '{0:05f}'.format(runData.percent_change[TRindex])
                file_name_to_save = getOutputFilename(run, TRindex)
                if cfg.mode == 'cloud':
                    full_filename_to_save = os.path.join(runFolderLinux, file_name_to_save) 
                else:
                    full_filename_to_save = os.path.join(runFolder, file_name_to_save) 
                # Send classification result back to the console computer
                fileInterface.putTextFile(full_filename_to_save, text_to_save)
                if args.commpipe:    
                    # JUST TO PLOT ON WEB SERVER
                    projUtils.sendResultToWeb(projComm, run, int(TRindex), runData.percent_change[TRindex])
            TRheader = makeTRHeader(cfg, runIndex, TRFilenum, TRindex, runData.percent_change[TRindex])
            TRindex += 1

        # TO DO - save the run regressor at each spot too maybe in the run folder, and maybe save specifically on the local display too

        # SAVE OVER RUN 
        runData.scanNum = scanNum # save scanning number
        runData.subjectName = cfg.subjectName
        runData.dicomDir = cfg.dicomDir
        run_filename = getRunFilename(cfg.sessionId, run)
        full_run_filename_to_save = os.path.join(runFolder, run_filename)
        #try:
        sio.savemat(full_run_filename_to_save, runData, appendmat=False)
        #except Exception as err:
        #    errorReply = self.createReplyMessage(msg, MsgResult.Errsor)
        #    errorReply.data = "Error: Unable to save blkGrpFile %s: %r" % (blkGrpFilename, err)
        #    return errorReply
    sys.exit(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()


