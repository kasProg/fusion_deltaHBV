# import sys
# sys.path.append('../../')
from hydroDL import master, utils
from hydroDL.data import camels
from hydroDL.master import default
from hydroDL.model import rnn, crit, train
from hydroDL.master import loadModel


import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import datetime as dt

import sys
from datetime import datetime

# fix the random seeds for reproducibility
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## GPU setting
# which GPU to use when having multiple
traingpuid = 6
torch.cuda.set_device(traingpuid)

# my_list = json.loads(sys.argv[1])
## Setting training options here
PUOpt = 0
# PUOpt values and explanations:
# 0: train and test on ALL basins;
# 1 for PUB spatial test, randomly hold out basins;
# 2 for PUR spatial test, hold out a continuous region;
buffOpt = 0
# buffOpt defines the warm-up option for the first year of training forcing data
# 0: do nothing, the first year forcing would only be used to warm up the next year;
# 1: repeat first year forcing to warm up the first year;3.988196838920481
# 2: load one more year forcing to warm up the first yearyear
TDOpt = True
# TDOpt, True as using dynamic parameters and False as using static parameters

multiforcing = True # set True if you want to use multiple forcings
if multiforcing == False:
    forType = 'nldas'
    # for Type defines which forcing in CAMELS to use: 'daymet', 'nldas', 'maurer'
else:
    # forType = ['daymet']
    forType = ['daymet', 'maurer_extended', 'nldas_extended']
    # forType = ['nldas_extended', 'maurer_extended']power
base_forType = 'maurer_extended'

load_hbv_path = "/data/kas7897/dPLHBVrelease/output/CAMELSDemo/dPLHBV_capillary/ALL" \
          "/TDTestforc/TD1_13/maurer_extended/BuffOpt0/NSE/111111/Fold1" \
          "/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/model_Ep50.pt"

#used only when multiforcing is True; else does not matter
prcp_loss_factor = 0
smooth_loss_factor = 0

## Set hyperparameters
EPOCH = 50 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 10
Ttrain = [19801001, 19951001] # Training period
# Ttrain = [19891001, 19991001] # PUB/PUR period
Tinv = [19801001, 19951001] # Inversion period for historical forcings
# Tinv = [19891001, 19991001] # PUB/PUR period
Nfea = 14 # number of HBV parameters. 12:original HBV; 13:includes the added dynamic ET para when setting ETMod=True #14 includes capillary
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = True # Whether to use the routing module for simulated runoff
Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correction

# Convert the date strings to datetime objects
dateTrain1 = datetime.strptime(str(Ttrain[0]), '%Y%m%d')
dateTrain2 = datetime.strptime(str(Ttrain[1]), '%Y%m%d')
delta_train = dateTrain2 - dateTrain1
num_days_train = delta_train.days


if TDOpt is True:
    # Below options are only for running models with dynamic parameters
    tdRep = [1, 13] # When using dynamic parameters, this list defines which parameters to set as dynamic
    tdRepS = [str(ix) for ix in tdRep]
    # ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
    # Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
    # If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
    ETMod = True
    Nfea = 14 # should be 13 when setting ETMod=True. 12 when ETMod=False
    dydrop = 0.0 # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
    staind = -1 # which time step to use from the learned para time series for those static parameters
    TDN = '/TDTestforc/'+'TD'+"_".join(tdRepS) +'/'
else:
    TDN = '/Testforc/'

# Define root directory of database and output
# Modify these based on your own location of CAMELS dataset
# Following the data download instruction in README file, you should organize the folders like
# 'your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow' and 'your/path/to/Camels/camels_attributes_v2.0'
# Then 'rootDatabase' here should be 'your/path/to/Camels';
# 'rootOut' is the root dir where you save the trained model
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory
# rootDatabase = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize camels module-scope variables in camels.py (dirDB, gageDict) to read basin info

# rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Model output root directory
rootOut = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease', 'output')  # Model output root directory

## set up different data loadings for ALL, PUB, PUR
testfoldInd = 1
# Which fold to hold out for PUB (10 folds, from 1 to 10) and PUR (7 folds, from 1 to 7).
# It doesn't matter when training on ALL basins (setting PUOpt=0), could always set testfoldInd=1 for this case.

# load CAMELS basin information
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
gageidLst = gageid.tolist()

if PUOpt == 0: # training on all basins without spatial hold-out
    puN = 'ALL'
    TrainLS = gageidLst # all basins
    TrainInd = [gageidLst.index(j) for j in TrainLS]
    TestLS = gageidLst
    TestInd = [gageidLst.index(j) for j in TestLS]
    gageDic = {'TrainID':TrainLS, 'TestID':TestLS}

elif PUOpt == 1: # random hold out basins. hold out the fold set by testfoldInd
    puN = 'PUB'
    # load the PUB basin groups
    # randomly divide CAMELS basins into 10 groups and this file contains the basin ID for each group
    # located in splitPath
    splitPath = 'PUBsplitLst.txt'
    with open(splitPath, 'r') as fp:
        testIDLst=json.load(fp)
    # Generate training ID lists excluding the hold out fold
    TestLS = testIDLst[testfoldInd - 1]
    TestInd = [gageidLst.index(j) for j in TestLS]
    TrainLS = list(set(gageid.tolist()) - set(TestLS))
    TrainInd = [gageidLst.index(j) for j in TrainLS]
    gageDic = {'TrainID':TrainLS, 'TestID':TestLS}

elif PUOpt == 2:
    puN = 'PUR'
    # Divide CAMELS dataset into 7 continous PUR regions, as shown in Feng et al, 2021 GRL; 2022 HESSD
    # get the id list of each PUR region, save to list
    regionID = list()
    regionNum = list()
    # seven regions including different HUCs
    regionDivide = [ [1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17] ]
    for ii in range(len(regionDivide)):
        tempcomb = regionDivide[ii]
        tempregid = list()
        for ih in tempcomb:
            tempid = gageid[hucinfo==ih].tolist()
            tempregid = tempregid + tempid
        regionID.append(tempregid)
        regionNum.append(len(tempregid))

    iexp = testfoldInd - 1  #index
    TestLS = regionID[iexp] # basin ID list for testing, hold out for training
    TestInd = [gageidLst.index(j) for j in TestLS]
    TrainLS = list(set(gageid.tolist()) - set(TestLS)) # basin ID for training
    TrainInd = [gageidLst.index(j) for j in TrainLS]
    gageDic = {'TrainID': TrainLS, 'TestID': TestLS}


# apply buffOPt to solve the warm-up for the first year
if buffOpt ==2: # load more BUFFTIME data for the first year
    sd = utils.time.t2dt(Ttrain[0]) - dt.timedelta(days=BUFFTIME)
    sdint = int(sd.strftime("%Y%m%d"))
    TtrainLoad = [sdint, Ttrain[1]]
    TinvLoad = [sdint, Ttrain[1]]
else:
    TtrainLoad = Ttrain
    TinvLoad = Tinv


# the attributes used to learn parameters
attrnewLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
               'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
               'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
               'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
               'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
               'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

attrWghts = ['p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
               'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
               'lai_diff', 'gvf_max', 'gvf_diff']

optData = default.optDataCamels # a default dictionary for logging, updated below
# Update the training period and variables

# if forType==['daymet', 'maurer_extended', 'nldas_extended']:
if type(forType) == list:
    #for all forcings
    # forcUN = np.empty([len(TrainInd), num_days_train, len(forType)*2])
    # forcInvUN = np.empty([len(TrainInd), num_days_train, len(forType)*2])

    #for multiple prcp only
    forcUN = np.empty([len(TrainInd), num_days_train, len(forType) + 1])
    forcInvUN = np.empty([len(TrainInd), num_days_train, len(forType) +1])
    # counter = 0
    for i in range(len(forType)):
        if forType[i] == 'daymet':
            varF = ['prcp', 'tmean']
            varFInv = ['prcp', 'tmean']
        else:
            varF = ['prcp', 'tmax']  # For CAMELS maurer and nldas forcings, tmax is actually tmean
            varFInv = ['prcp', 'tmax']

        if 'daymet' in forType:
            optData = default.update(optData, tRange=TtrainLoad, varT=varFInv, varC=attrnewLst, subset=TrainLS,
                                     forType='daymet')
        elif 'nldas' in forType:
            optData = default.update(optData, tRange=TtrainLoad, varT=varFInv, varC=attrnewLst, subset=TrainLS,
                                     forType='nldas')
        elif 'nldas_extended' in forType:
            optData = default.update(optData, tRange=TtrainLoad, varT=varFInv, varC=attrnewLst, subset=TrainLS,
                                     forType='nldas_extended')
        else:
            optData = default.update(optData, tRange=TtrainLoad, varT=varFInv, varC=attrnewLst, subset=TrainLS,
                                     forType=forType[0])

        dfTrain = camels.DataframeCamels(tRange=TtrainLoad, subset=TrainLS, forType=forType[i])
        forcUN_type = dfTrain.getDataTs(varLst=varF, doNorm=False, rmNan=False)

        dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType[i])
        forcInvUN_type = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)

        forcUN[:, :, i] = forcUN_type[:, :, 0]
        forcInvUN[:, :, i] = forcInvUN_type[:, :, 0]
        forcUN[:, :, -1] = forcUN_type[:, :, 1]
        forcInvUN[:, :, -1] = forcInvUN_type[:, :, 1]
        if forType[i] == 'daymet':
            daymet_temp = forcUN_type[:, :, 1]
            daymetInV_temp = forcInvUN_type[:, :, 1]
        if forType[i] == 'nldas' or forType[i] == 'nldas_extended':
            nldas_temp = forcUN_type[:, :, 1]
            nldasInV_temp = forcInvUN_type[:, :, 1]

        #for all forcings
        # forcUN[:,:,i] = forcUN_type[:,:,0]
        # forcUN[:,:,i+3] = forcUN_type[:,:,1]
        # forcInvUN[:,:,i] = forcInvUN_type[:,:,0]
        # forcInvUN[:,:,i+3] = forcInvUN_type[:,:,1]


    obsUN = dfTrain.getDataObs(doNorm=False, rmNan=False, basinnorm=False)
    attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)
    attrs_wghtsUN = dfInv.getDataConst(varLst=attrWghts, doNorm=False, rmNan=False)

    if 'daymet' in forType:
        forcUN[:, :, -1] = daymet_temp
        forcInvUN[:, :, -1] = daymetInV_temp
    elif 'nldas' in forType or 'nldas_extended' in forType:
        forcUN[:, :, -1] = nldas_temp
        forcInvUN[:, :, -1] = nldasInV_temp




if base_forType == 'daymet':
    varFInv_base = ['prcp', 'tmean']
else:
    varFInv_base = ['prcp', 'tmax']
dfInv_base = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=base_forType)
forcInvUN_base = dfInv_base.getDataTs(varLst=varFInv_base, doNorm=False, rmNan=False)


# for HBV model training inputs


# dfInv = camels.DataframeCamels(tRange=TinvLoad, subset=TrainLS, forType=forType)
# forcInvUN = dfInv.getDataTs(varLst=varFInv, doNorm=False, rmNan=False)
# attrsUN = dfInv.getDataConst(varLst=attrnewLst, doNorm=False, rmNan=False)

# Unit transformation, discharge obs from ft3/s to mm/day
areas = gageinfo['area'][TrainInd] # unit km2
temparea = np.tile(areas[:, None, None], (1, obsUN.shape[1],1))
obsUN = (obsUN * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10**3 # transform to mm/day

# load potential ET calculated by hargreaves method
varLstNL = ['PEVAP']
usgsIdLst = gageid

#for multiple PETs
# PETUN = np.empty([len(usgsIdLst), num_days_train, len(forType)])
# PETInvUN = np.empty([len(usgsIdLst), num_days_train, len(forType)])
# if type(forType) == list:
#     for i in range(len(forType)):
#         if forType[i] == 'nldas_extended' or forType[i] == 'nldas':
#             PETDir = rootDatabase + '/pet_harg/' + 'nldas' + '/'
#             tPETRange = [19800101, 20150101]
#             tPETLst = utils.time.tRange2Array(tPETRange)
#         if forType[i] == 'maurer_extended' or forType[i] == 'maurer':
#             PETDir = rootDatabase + '/pet_harg/' + 'maurer' + '/'
#             tPETRange = [19800101, 20090101]
#             tPETLst = utils.time.tRange2Array(tPETRange)
#         if forType[i] == 'daymet':
#             PETDir = rootDatabase + '/pet_harg/' + 'daymet' + '/'
#             tPETRange = [19800101, 20150101]
#             tPETLst = utils.time.tRange2Array(tPETRange)
#         ntime = len(tPETLst)
#         PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
#         for k in range(len(usgsIdLst)):
#             dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
#             PETfull[k, :, :] = dataTemp
#         TtrainLst = utils.time.tRange2Array(TtrainLoad)
#         TinvLst = utils.time.tRange2Array(TinvLoad)
#         C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
#         PETUN_type = PETfull[:, ind2, :]
#         PETUN_type = PETUN_type[TrainInd, :, :] # select basins
#         PETUN[:,:,i]  = PETUN_type[:,:,0]
#         C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
#         PETInvUN_type = PETfull[:, ind2inv, :]
#         PETInvUN_type = PETInvUN_type[TrainInd, :, :]
#         PETInvUN[:,:,i] = PETInvUN_type[:,:,0]

if forType == 'maurer' or forType=='maurer_extended' or forType == ['maurer'] or forType == ['maurer_extended']:
    tPETRange = [19800101, 20090101]
else:
    tPETRange = [19800101, 20150101]
tPETLst = utils.time.tRange2Array(tPETRange)
# Modify this as the directory where you put PET
if type(forType) == list:
    if forType[0]=='nldas_extended':
        PETDir = rootDatabase + '/pet_harg/' + 'nldas' + '/'
    elif forType[0]=='maurer_extended':
        PETDir = rootDatabase + '/pet_harg/' + 'maurer' + '/'
    else:
        PETDir = rootDatabase + '/pet_harg/' + forType[0] + '/'
else:
    PETDir = rootDatabase + '/pet_harg/' + forType + '/'

ntime = len(tPETLst)
PETfull = np.empty([len(usgsIdLst), ntime, len(varLstNL)])
for k in range(len(usgsIdLst)):
    dataTemp = camels.readcsvGage(PETDir, usgsIdLst[k], varLstNL, ntime)
    PETfull[k, :, :] = dataTemp

TtrainLst = utils.time.tRange2Array(TtrainLoad)
TinvLst = utils.time.tRange2Array(TinvLoad)
C, ind1, ind2 = np.intersect1d(TtrainLst, tPETLst, return_indices=True)
PETUN = PETfull[:, ind2, :]
PETUN = PETUN[TrainInd, :, :] # select basins
C, ind1, ind2inv = np.intersect1d(TinvLst, tPETLst, return_indices=True)
PETInvUN = PETfull[:, ind2inv, :]
PETInvUN = PETInvUN[TrainInd, :, :]

# process data, do normalization and remove nan
series_inv = np.concatenate([forcInvUN, PETInvUN], axis=2)
series_inv_hbv = np.concatenate([forcInvUN_base, PETInvUN], axis=2)
seriesvarLst = varFInv + ['pet']
# calculate statistics for normalization and saved to a dictionary
statDict_hbv = camels.getStatDic(attrLst=attrnewLst, attrdata=attrsUN, seriesLst=seriesvarLst, seriesdata=series_inv_hbv)
statDict_wghts = camels.getStatDic(attrLst=attrWghts, attrdata=attrs_wghtsUN, seriesLst=['prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'tmax', 'pet'], seriesdata=series_inv)
# normalize data
attr_norm = camels.transNormbyDic(attrsUN, attrnewLst, statDict_hbv, toNorm=True)
attrWghts_norm = camels.transNormbyDic(attrs_wghtsUN, attrWghts, statDict_wghts, toNorm=True)
attr_norm[np.isnan(attr_norm)] = 0.0
attrWghts_norm[np.isnan(attrWghts_norm)] = 0.0
series_norm_hbv = camels.transNormbyDic(series_inv_hbv, seriesvarLst, statDict_hbv, toNorm=True)
series_Wghts_norm = camels.transNormbyDic(series_inv, ['prcp_daymet', 'prcp_maurer', 'prcp_nldas', 'tmax', 'pet'], statDict_wghts, toNorm=True)
series_norm_hbv[np.isnan(series_norm_hbv)] = 0.0
series_Wghts_norm[np.isnan(series_Wghts_norm)] = 0.0

# prepare the inputs
zTrain_hbv = series_norm_hbv # used as the inputs for dPL inversion gA along with attributes
zTrain_wghts = series_Wghts_norm # used as the inputs for dPL inversion gA along with attributes
xTrain_wghts = np.concatenate([forcUN, PETUN], axis=2) # used as HBV forcing
xTrain_hbv = xTrain_wghts[:,:,(0,-2,-1)] # used as HBV forcing
xTrain_wghts[np.isnan(xTrain_wghts)] = 0.0
xTrain_hbv[np.isnan(xTrain_hbv)] = 0.0

if buffOpt == 1: # repeat the first year warm up the first year itself
    zTrainIn_hbv = np.concatenate([zTrain_hbv[:,0:BUFFTIME,:], zTrain_hbv], axis=1)
    zTrainIn_wghts = np.concatenate([zTrain_wghts[:,0:BUFFTIME,:], zTrain_wghts], axis=1)
    xTrainIn_wghts = np.concatenate([xTrain_wghts[:,0:BUFFTIME,:], xTrain_wghts], axis=1) # repeat forcing to warm up the first year
    xTrainIn_hbv = np.concatenate([xTrain_hbv[:,0:BUFFTIME,:], xTrain_hbv], axis=1) # repeat forcing to warm up the first year
    yTrainIn = np.concatenate([obsUN[:,0:BUFFTIME,:], obsUN], axis=1)
else: # no repeat, original data, the first year data would only be used as warmup for the next following year
    zTrainIn_hbv = zTrain_hbv
    zTrainIn_wghts = zTrain_wghts
    xTrainIn_wghts = xTrain_wghts
    xTrainIn_hbv = xTrain_hbv
    yTrainIn = obsUN

forcTuple_hbv = (xTrainIn_hbv, zTrainIn_hbv)
forcTuple_wghts = (xTrainIn_wghts, zTrainIn_wghts)
attrs = attr_norm
attrs_wghts = attrWghts_norm

## Train the model
# define loss function
alpha = 0.25 # a weight for RMSE loss to balance low and peak flow
# optLoss = default.update(default.optLossComb, name='hydroDL.model.crit.RmseLossComb', weight=alpha)
optLoss = default.update(default.optLossNSE, name='hydroDL.model.crit.NSELossBatch')
# lossFun = crit.RmseLossComb(alpha=alpha)
lossFun = crit.NSELossBatch(np.nanstd(yTrainIn, axis=1))

# define training options
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH)
# define output folder to save model results
exp_name = 'CAMELSDemo'
if forType==['daymet', 'maurer_extended', 'nldas_extended']:
    exp_disp = 'LSTM-dPLHBV/' + puN + TDN + 'allprcp_36withloss' + str(prcp_loss_factor) + 'smooth' + str(smooth_loss_factor) + '/BuffOpt'+str(buffOpt)+'/RMSE_para'+str(alpha)+'/' + str(randomseed) + \
           '/Fold' + str(testfoldInd)
elif forType==['daymet', 'maurer', 'nldas']:
    exp_disp = 'dPLHBV/' + puN + TDN + 'all_withloss' + str(prcp_loss_factor) + 'smooth' + str(smooth_loss_factor) + '/BuffOpt'+str(buffOpt)+'/RMSE_para'+str(alpha)+'/' + str(randomseed) + \
           '/Fold' + str(testfoldInd)
elif type(forType)==list:
    forType_string = '|'.join(forType)
    exp_disp = 'dPLHBV/' + puN + TDN + forType_string + 'withloss' + str(prcp_loss_factor)+ 'smooth' + str(smooth_loss_factor) + '/BuffOpt'+str(buffOpt)+'/RMSE_para'+str(alpha)+'/' + str(randomseed) + \
           '/Fold' + str(testfoldInd)
else:
    exp_disp = 'dPLHBV/' + puN + TDN + forType + '/BuffOpt'+str(buffOpt)+'/RMSE_para'+str(alpha)+'/' + str(randomseed) + \
            '/Fold' + str(testfoldInd)
exp_info = 'T_'+str(Ttrain[0])+'_'+str(Ttrain[1])+'_BS_'+str(BATCH_SIZE)+'_HS_'+str(HIDDENSIZE)\
           +'_RHO_'+str(RHO)+'_NF_'+str(Nfea)+'_Buff_'+str(BUFFTIME)+'_Mul_'+str(Nmul)
save_path = os.path.join(exp_name, exp_disp)
out = os.path.join(rootOut, save_path, exp_info) # output folder to save results
# define and load model
Ninv = zTrain_wghts.shape[-1] + attrs_wghts.shape[-1]

if TDOpt is False:
    # model with all static parameters
    model = rnn.prcp_weights(ninv=Ninv, hiddeninv=HIDDENSIZE, prcp_datatypes=len(forType))
    # dict only for logging
    optModel = OrderedDict(name='LSTM-dPLHBV', nx=Ninv, nfea=Nfea, nmul=Nmul, hiddenSize=HIDDENSIZE, doReLU=True,
                           Tinv=Tinv, Trainbuff=BUFFTIME, routOpt=routing, comprout=comprout, compwts=compwts,
                           pcorr=pcorr, buffOpt=buffOpt, TDOpt=TDOpt)
else:
    # model with some dynamic parameters
    model = rnn.prcp_weights(ninv=Ninv, hiddeninv=HIDDENSIZE, prcp_datatypes=len(forType))
    # dict only for logging
    optModel = OrderedDict(name='LSTM-dPLHBV', nx=Ninv, nfea=Nfea, nmul=Nmul, hiddenSize=HIDDENSIZE, doReLU=True,
                           Tinv=Tinv, Trainbuff=BUFFTIME, routOpt=routing, comprout=comprout, compwts=compwts,
                           pcorr=pcorr, staind=staind, tdlst=tdRep, dydrop=dydrop,buffOpt=buffOpt, TDOpt=TDOpt, ETMod=ETMod)



# Wrap up all the training configurations to one dictionary in order to save into "out" folder as logging
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
master.writeMasterFile(masterDict)
# log statistics for normalization
statFile_wghts = os.path.join(out, 'statDict_wghts.json')
with open(statFile_wghts, 'w') as fp:
    json.dump(statDict_wghts, fp, indent=4)
statFile_hbv = os.path.join(out, 'statDict_hbv.json')
with open(statFile_hbv, 'w') as fp:
    json.dump(statDict_hbv, fp, indent=4)



loaded_hbv = torch.load(load_hbv_path, map_location=torch.device(f"cuda:{traingpuid}"))

# Train the model
trainedModel = train.train2Model(
    model,
    loaded_hbv,
    forcTuple_wghts,
    forcTuple_hbv,
    yTrainIn,
    attrs_wghts,
    attrs,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME,
    multiforcing=multiforcing,
    prcp_loss_factor=prcp_loss_factor,
    smooth_loss_factor=smooth_loss_factor)
