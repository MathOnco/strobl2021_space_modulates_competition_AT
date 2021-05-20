# ====================================================================================
# Script to perform the separation experiment shown in Figure 4. Extracts the ttp from the CA data
# and computes the relative and absolute gains of AT. In addition, it keeps track of the
# competition metrics.
#  ====================================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import os
import multiprocessing as mp

# ================================ Script setup ==========================================
sweepToPerform = "separationExperimentSweep" # Which sweep to perform. For details see below
domainSize = 100
initialDensityList = [0.5]
rFracList = [.01] # meaningless here
nReplicates = 1000
tEnd = 3.65e3
dt = 1.
nProcesses = 14
outDir = "/scratch/strobl/data/separationExperiment1d"

if sweepToPerform == "separationExperimentSweep": # Sweep over a fine grid of different distances. Used in Figure 5b
    costList = [0.]
    turnoverList = [0.]
    distList = np.arange(0,50,8)
    outName = "./separationExperimentSweep.csv"
elif sweepToPerform == "separationExperimentSweep_costTurnover": # Sweep over cost and turnover values. Used in Figure 5e
    costList = [0.,0.1,0.2,0.3]
    turnoverList = [0.,0.05,0.1,0.3]
    distList = [0,8,16,32]
    outName = "./separationExperimentSweep_costTurnover.csv"
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
            dataToDetermineTTP = tmpDf.tail(5)
            ttpArr[i] = dataToDetermineTTP[dataToDetermineTTP.NCells > 1.2 * initialSize * paramDic['xDim']**2].Time.min()
            finalRSizeArr[i] = tmpDf.NCells_R.iloc[-1]
            tmpDf = tmpDf[tmpDf.Time<ttpArr[0]]
            tmpDf['PropFailedDivs'] = tmpDf.NFailedDivs/tmpDf.NAttemptedDivs*100
            # tmpDf['PerCapitaGrowthRate'] = ((tmpDf.NAttemptedDivs-tmpDf.NFailedDivs)-tmpDf.NDeaths)/tmpDf.NCells_R
            tmpDfList.append(tmpDf)
            os.remove(currDfName)

        # Compute the competition differential metric
        dataDf = pd.concat(tmpDfList)
        # dataDf['PropFailedDivs'] = dataDf.NFailedDivs / dataDf.NAttemptedDivs
        diffVec = dataDf.PropFailedDivs[dataDf.TreatmentBranch == "AT50"] - dataDf.PropFailedDivs[
            dataDf.TreatmentBranch == "MTD"]
        diffVec = diffVec[np.isnan(diffVec) == False]

        # Obtain the relative time gained in the CA
        relTimeGained_CA = (ttpArr[1] - ttpArr[0]) / ttpArr[0]
        tmpDicList.append({"RFrac": rFrac, "InitialTumourSize": initialSize, "Cost": cost,
                           "Turnover": paramDic['turnover'],
                           "ReplicateId": replicateId,
                           "Distance":paramDic['initialSeedingDistance'], "DomainSize":paramDic['xDim'],
                           "TTP_CT_CA": ttpArr[0], "TTP_AT50_CA": ttpArr[1],
                           "RelTimeGained_CA": relTimeGained_CA * 100,
                           "MaxNCycles": tmpDf.NCycles.max(),
                           "FinalSize_R_CT": finalRSizeArr[0],
                           "FinalSize_R_AT": finalRSizeArr[1],
                           "SuppressionDiff_Abs": np.linalg.norm(diffVec, ord=1),
                           "SuppressionDiff_Sum": np.sum(diffVec),
                           "SuppressionDiff_Mean": np.mean(diffVec),
                           "SuppressionDiff_Sq": np.linalg.norm(diffVec, ord=2)})

    # 3. Save
    return pd.DataFrame(tmpDicList)
# ================================= Main ================================================
pool = mp.Pool(processes=nProcesses)

# 1. Run the simulations
jobList = []
for initialDensity, rFrac, cost, turnover, distance in product(initialDensityList, rFracList, costList, turnoverList, distList):
    currOutDir = os.path.join(outDir, "distance%d/cost%d/turnover%d/" % (distance,cost*100,turnover*100))
    jobList.append({"initialSize":initialDensity, "rFrac":rFrac, "turnover":turnover, "cost":cost,
                    "xDim":domainSize,"yDim":domainSize,
                    "initialSeedingType": "separation1d", "initialSeedingDistance": distance,
                    "tEnd":tEnd,"nReplicates":nReplicates,"outDir":currOutDir,"profilingMode":"false"})

# print(len(jobList))
tmpDfList = list(tqdm(pool.imap(RunSimulation, jobList), total=len(jobList)))

# 2. Collect the data
tmpDf = pd.concat(tmpDfList)
tmpDf.to_csv(outName)
# os.rmdir(outDir)