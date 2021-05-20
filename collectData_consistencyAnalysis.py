# ====================================================================================
# Script to collect the data for the consistency analysis. Extracts the ttp from the CA data
# and computes the relative and absolute gains of AT.
# ====================================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import os
import multiprocessing as mp

# ================================ Script setup ==========================================
initialSizeList = [0.25, 0.75]
rFracList = [.001, 0.1]
turnoverList = [0.3]
costList = [0]
nReplicates = 34100
tEnd = 3.65e3
dt = 1.
nProcesses = 4
outDir = "/scratch/strobl/data/consistencyAnalysis/" # "data/scrap/consistencyAnalysis" #
outName = "consistencyAnalysis.csv" # "./consistencyAnalysis.csv" #
# ============================= Auxillary Functions ======================================
def RunSimulation(paramDic):
    argStr = " "
    for var in paramDic.keys():
        argStr += "-%s %s "%(var,paramDic[var])
    os.system("java -jar onLatticeModel.jar"+argStr)
    return ProcessData(paramDic)

def ProcessData(paramDic):
    initialSize, rFrac, cost, turnover = (paramDic['initialSize'],paramDic['rFrac'],paramDic['cost'],paramDic['turnover'])
    tmpDicList = []
    # 1. Compute gain in TTP
    for replicateId in tqdm(range(nReplicates),disable=True):
        ttpArr = [np.nan, np.nan]
        finalRSizeArr = [np.nan, np.nan]
        tmpDfList = []
        for i, txName in enumerate(["MTD", "AT50"]):
            currDfName = os.path.join(paramDic['outDir'],
                                      "%s_cellCounts_cost_%1.1f_rFrac_%.2g_initSize_%.2g_dt_%1.1f_RepId_%d.csv" % (
                                          txName, cost * 100, rFrac, initialSize, dt, replicateId))
            tmpDf = pd.read_csv(currDfName)
            tmpDf['TreatmentBranch'] = txName
            dataToDetermineTTP = tmpDf[tmpDf.Time>tmpDf.Time.max()/2]
            ttpArr[i] = dataToDetermineTTP[dataToDetermineTTP.NCells > 1.2 * initialSize * 1e4].Time.min()
            finalRSizeArr[i] = tmpDf.NCells_R.iloc[-1]
            tmpDfList.append(tmpDf)
            os.remove(currDfName)

        # Compute the competition differential metric
        dataDf = pd.concat(tmpDfList)
        dataDf['PropFailedDivs'] = dataDf.NFailedDivs / dataDf.NAttemptedDivs
        diffVec = dataDf.PropFailedDivs[dataDf.TreatmentBranch == "MTD"] - dataDf.PropFailedDivs[
            dataDf.TreatmentBranch == "AT50"]
        diffVec = diffVec[np.isnan(diffVec) == False]

        # Obtain the relative time gained in the CA
        relTimeGained_CA = (ttpArr[1] - ttpArr[0]) / ttpArr[0]
        tmpDicList.append({"RFrac": rFrac, "InitialTumourSize": initialSize, "Cost": cost,
                           "Turnover": paramDic['turnover'], "ReplicateId": replicateId,
                           "TTP_CT_CA": ttpArr[0], "TTP_AT50_CA": ttpArr[1],
                           "RelTimeGained_CA": relTimeGained_CA * 100,
                           "MaxNCycles": tmpDf.NCycles.max(),
                           "FinalSize_R_CT": finalRSizeArr[0],
                           "FinalSize_R_AT": finalRSizeArr[1],
                           "SuppressionDiff_Abs": np.linalg.norm(diffVec, ord=1),
                           "SuppressionDiff_Sq": np.linalg.norm(diffVec, ord=2)})
    # 3. Save
    return pd.DataFrame(tmpDicList)

# ================================= Main ================================================
pool = mp.Pool(processes=nProcesses)

# 1. Run the simulations
jobList = []
for initialSize, rFrac, cost, turnover, in product(initialSizeList, rFracList, costList, turnoverList):
    currOutDir = os.path.join(outDir,"turnover%d/cost%d/"%(turnover*100,cost*100))
    jobList.append({"initialSize":initialSize, "rFrac":rFrac, "turnover":turnover, "cost":cost,
                    "tEnd":tEnd,"nReplicates":nReplicates,"outDir":currOutDir,"profilingMode":"false"})

# print(len(jobList))
tmpDfList = list(tqdm(pool.imap(RunSimulation, jobList), total=len(jobList)))

# 2. Collect the data
tmpDf = pd.concat(tmpDfList)
tmpDf.to_csv(outName)

# 3. Name each tumour and assign each replicate to one of the distributions (defined by the sample number in the group and the group id)
nSamplesList = [10,50,100,250,500,1000,1500]
nGroupsPerSampleSize = 10

def NameTumours(initialSize,rFrac):
    if initialSize == 0.25:
        if rFrac == 0.001:
            return 1
        else:
            return 2
    else:
        if rFrac == 0.1:
            return 3
        else:
            return 4

tmpDfList = []
for initialSize,rFrac in product(initialSizeList,rFracList):
    currDataDf = tmpDf[(tmpDf.InitialTumourSize==initialSize) &
                       (tmpDf.RFrac==rFrac)].copy()
    tumourId = NameTumours(initialSize, rFrac)
    currDataDf['TumourId'] = tumourId
    currDataDf['GroupId'] = np.nan
    currDataDf['NSamples'] = np.nan
    replicateIdList = currDataDf.ReplicateId.unique()
    np.random.shuffle(replicateIdList)  # Shuffle list to get random group assignment
    k = 0
    for nSamples, groupId in product(nSamplesList, range(nGroupsPerSampleSize)):
        for _ in range(nSamples):
            currDataDf.loc[k, 'NSamples'] = nSamples
            currDataDf.loc[k, 'GroupId'] = groupId
            k += 1
    tmpDfList.append(currDataDf)
tmpDf = pd.concat(tmpDfList)
tmpDf.to_csv(outName)