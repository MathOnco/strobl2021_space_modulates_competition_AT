# ====================================================================================
# Script to collect data on the relationship between nest convexity (ruggedness) and
# outcome.
# ====================================================================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import os
import shutil
import multiprocessing as mp
import cv2
from scipy.spatial import distance

# ================================ Script setup ==========================================
initialSizeList = [0.25]
rFracList = [0.001]
costList = [0.]
turnoverList = [0.]
nReplicates = 1000
imageFreq = 50
timePointList = [150,250]
tEnd = 2.6e2
dt = 1.
nProcesses = 2
outDir = "./data_nestCounting/"
outName = "./convexityAnalysis.csv"
# ============================= Auxillary Functions ======================================
def RunSimulation(paramDic):
    argStr = " "
    for var in paramDic.keys():
        argStr += "-%s %s "%(var,paramDic[var])
    os.system("java -jar onLatticeModel.jar"+argStr)

def PerformImageAnalysis(jobDic):
    initialSize, rFrac, turnover, cost = [jobDic['initialSize'],jobDic['rFrac'],jobDic['turnover'],jobDic['cost']]
    tmpDicList = []
    imgDir = os.path.join(outDir, "tmp/")
    paramDic = {"initialSize": initialSize, "rFrac": rFrac, "turnover": turnover, "cost": cost,
                "tEnd": tEnd, "nReplicates": nReplicates, "outDir": outDir,
                "imageOutDir": imgDir, "imageFreq": imageFreq}
    RunSimulation(paramDic)

    # 2. Analyse the images
    for replicateId in range(nReplicates):
        # a) Compute TTP
        ttpArr = [np.nan, np.nan]
        finalRSizeArr = [np.nan, np.nan]
        for i, txName in enumerate(["MTD", "AT50"]):
            currDfName = os.path.join(outDir,
                                      "%s_cellCounts_cost_%1.1f_rFrac_%.2g_initSize_%.2g_dt_%1.1f_RepId_%d.csv" % (
                                          txName, cost * 100, rFrac, initialSize, dt, replicateId))
            tmpDf = pd.read_csv(currDfName)
            ttpArr[i] = tmpDf[tmpDf.NCells > 1.2 * initialSize * 1e4].Time.min()
            finalRSizeArr[i] = tmpDf.NCells_R.iloc[0]

            os.remove(currDfName)
        # Obtain the relative time gained in the CA
        absTimeGained_CA = ttpArr[1] - ttpArr[0]
        relTimeGained_CA = ((absTimeGained_CA) / ttpArr[0]) * 100

        # b) Analyse images
        for j, txName in enumerate(["MTD", "AT50"]):
            for timePoint in timePointList:
                # Load image
                currImgDir = os.path.join(imgDir, "%s_cost_%1.1f_rFrac_%.2g_initSize_%.2g_dt_%1.1f_RepId_%d" % (txName, cost * 100, rFrac, initialSize, dt, replicateId))
                convexityList = []
                meanDist = np.nan
                if os.path.isfile(os.path.join(currImgDir, "img_t_%.1f.png" % (float(timePoint)))):
                    currImg = cv2.imread(os.path.join(currImgDir, "img_t_%.1f.png" % (float(timePoint))))
                    currImg[currImg[:, :, 2] == 117, :] = [0, 0, 0]  # Remove sensitive cells
                    currImg = cv2.cvtColor(currImg, cv2.COLOR_RGB2GRAY)

                    # 1. Identify the number of nests
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(currImg)
                    nNests = num_labels-1

                    # 2. Compute the mean distance between nests at time 0
                    if timePoint==0:
                        distMat = distance.cdist(centroids[1:], centroids[1:], 'euclidean')
                        sumDistances = 0
                        for k in range(nNests):
                            sumDistances += np.sum(distMat[k:, k])
                        meanDist = sumDistances / (nNests * (nNests - 1) / 2)

                    # 3. Compute the convexity
                    for nestId in range(1, nNests + 1):
                        currNest = labels.copy()
                        currNest[currNest != nestId] = 0
                        # Find the boundary of the nest which will be the largest contour
                        contours = cv2.findContours(currNest, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
                        nestContour = contours[0][np.array([len(c) for c in contours[0]]).argmax()]

                        # Compute its convexity
                        hull = cv2.convexHull(nestContour, returnPoints=True)
                        perimeter_nest = cv2.arcLength(nestContour, closed=True)
                        perimeter_convexHull = cv2.arcLength(hull, closed=True)
                        convexityList.append(perimeter_convexHull / perimeter_nest)
                else:
                    nNests = np.nan
                tmpDicList.append({"RFrac": rFrac, "InitialTumourSize": initialSize, "Cost": cost,
                                   "Turnover": paramDic['turnover'], "ReplicateId": replicateId,
                                   "Time": timePoint, "TreatmentBranch": txName,
                                   "TTP_CT_CA": ttpArr[0], "TTP_AT50_CA": ttpArr[1],
                                   "AbsTimeGained_CA": absTimeGained_CA,
                                   "RelTimeGained_CA": relTimeGained_CA,
                                   "MaxNCycles": tmpDf.NCycles.max(),
                                   "FinalSize_R_CT": finalRSizeArr[0],
                                   "FinalSize_R_AT": finalRSizeArr[1],
                                   "NNests": nNests, "InitialMeanDist":meanDist,
                                   "MeanConvexity":np.mean(convexityList)})
            # Clean up
            shutil.rmtree(currImgDir, ignore_errors=True)
    return pd.DataFrame(tmpDicList)
# ================================= Main ================================================
pool = mp.Pool(processes=nProcesses,maxtasksperchild=1)

# 1. Run the simulations
jobList = []
# tmpDfList = []
for initialSize, rFrac, cost, turnover in product(initialSizeList,rFracList,costList,turnoverList):
    jobList.append({'initialSize':initialSize,'rFrac':rFrac,'turnover':turnover,'cost':cost,'seedList':np.arange(0,1000)})
    # PerformImageAnalysis(jobList[-1])

tmpDfList = list(tqdm(pool.imap(PerformImageAnalysis, jobList),total=len(jobList)))
tmpDf = pd.concat(tmpDfList)
tmpDf.to_csv(outName)
shutil.rmtree(outDir, ignore_errors=True)