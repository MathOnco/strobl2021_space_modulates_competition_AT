# ====================================================================================
# Script to collect the data to compare the carrying capacity seen in the CA to that seen
# in the ODE model.
# ====================================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import os
import multiprocessing as mp

# ================================ Script setup ==========================================
# initialSizeList = [0.25] #0.35, 0.45, 0.55, 0.65
# rFracList = [0.001]
# costList = [np.round(x,2) for x in np.linspace(0,0.5,11)]
# turnoverList = [np.round(x,2) for x in np.linspace(0,0.5,11)]
initialSizeList = [0.5]
rFracList = [0.01]
costList = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
turnoverList = [0,0.1,0.3]
nReplicates = 1000
tEnd = 3.65e3
dt = 1.
nProcesses = 18
outDir = "/scratch/strobl/data/effectiveKData/"
outName = "./effectiveKData.csv"
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
    txName = "AT0"
    tmpDicList = []
    for replicateId in tqdm(range(nReplicates),disable=True):
        currDfName = os.path.join(paramDic['outDir'],
                                  "%s_cellCounts_cost_%1.1f_rFrac_%1.1f_initSize_%.2g_dt_%1.1f_RepId_%d.csv" % (
                                      txName, cost * 100, rFrac, initialSize, dt, replicateId))
        tmpDf = pd.read_csv(currDfName)
        tmpDicList.append({"RFrac": rFrac, "InitialTumourSize": initialSize, "Cost": cost,
                           "Turnover": paramDic['turnover'], "ReplicateId": replicateId,
                           "SteadyStateSize":tmpDf.NCells_R.iloc[-1]})
        os.remove(currDfName)

    # 3. Save
    return pd.DataFrame(tmpDicList)
# ================================= Main ================================================
pool = mp.Pool(processes=nProcesses)

# 1. Run the simulations
jobList = []
for initialSize, rFrac, cost, turnover, in product(initialSizeList, rFracList, costList, turnoverList):
    currOutDir = os.path.join(outDir,"turnover%d/cost%d/"%(turnover*100,cost*100))
    jobList.append({"initialSize": rFrac * initialSize, "rFrac": 1., "turnover": turnover, "cost": cost,
                  "tEnd": tEnd, "nReplicates": nReplicates, "atThreshold": 0.,
                  "compareToMTD": False,
                  "profilingMode": "false", "terminateAtProgression": "false",
                  "outDir": currOutDir})

# for job in jobList[10:]:
#     RunSimulation(job)
# # print(len(jobList))
tmpDfList = list(tqdm(pool.imap(RunSimulation, jobList), total=len(jobList)))

# 2. Collect the data
tmpDf = pd.concat(tmpDfList)
tmpDf.to_csv(outName)
# os.rmdir(outDir)