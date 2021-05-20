# ====================================================================================
# Various functions that I found useful in this project
# ====================================================================================
import numpy as np
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from tqdm import tqdm
import string
import datetime
import os

# ====================================================================================
# Plotting
# ====================================================================================
# Function to plot growth data
def PlotData(growthData, nDaysToFit=-1,**kwargs):
    if nDaysToFit<0: nDaysToFit = growthData['Time'].max()
    spheroidId = growthData.SpheroidId.unique()[0]
    # Fitted data points
    lnslist = []
    lnslist += plt.plot(growthData['Time'][growthData['Time'] < nDaysToFit],
             growthData.FluorescentArea[growthData['Time'] < nDaysToFit],
             linestyle="None", marker=kwargs.get('markerTypeTrD', "x"), markersize=kwargs.get('markerSizeTrD', 10),
             color="black", markeredgewidth=2, label="Training Data")
    # Observed future growth
    lnslist += plt.plot(growthData['Time'][growthData['Time'] > nDaysToFit],
             growthData.FluorescentArea[growthData['Time'] > nDaysToFit],
             linestyle="None", marker=".", color="green", markersize=kwargs.get('markerSizeTeD', 20),
             label="Testing Data")

    # Plot the drug concentration
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    drugConcentrationVec = TreatmentListToTS(treatmentList=ExtractTreatmentFromDf(growthData),
                                             tVec=growthData['Time'])
    drugLn = ax2.fill_between(growthData['Time'],
                     0, drugConcentrationVec, color="#8f59e0", alpha=0.2, label="Drug Concentration")
    # Format the plot
    plt.xlim([0, kwargs.get('xlim', 1.1*growthData['Time'].max())])
    ax1.set_ylim([kwargs.get('yMin',0), kwargs.get('yLim', 1.1*growthData.FluorescentArea.max())])
    ax2.set_ylim([0, kwargs.get('y2lim', max(1.1*growthData.DrugConcentration.max(),.1))])
    ax1.set_xlabel("Days since start of the experiment")
    ax1.set_ylabel("Fluorescent Area")
    ax2.set_ylabel(r"Drug Concentration in $\mu M$")
    plt.title(kwargs.get('titleStr',''))
    if nDaysToFit<growthData['Time'].max(): plt.vlines(x=nDaysToFit, ymin=0, ymax=5e5, linestyle="--")
    if kwargs.get('plotLegendB', True):
        labsList = [l.get_label() for l in lnslist]
        plt.legend(lnslist, labsList, loc=kwargs.get('legendLoc', "upper left"))
    plt.tight_layout()
    if kwargs.get('saveFigB', False):
        plt.savefig(kwargs.get('outName', 'growthData_%s.png'%spheroidId), orientation='portrait', format='png')
    if kwargs.get('returnAx', False): return (ax1, ax2)
# # Function to plot growth data
# def PlotData(growthData, nDaysToFit=-1, **kwargs):
#     if nDaysToFit<0: nDaysToFit = growthData['Time'].max()
#     spheroidId = growthData.SpheroidId.unique()[0]
#     # Fitted data points
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx() # instantiate a second axis to hold the drug information
#     lnslist = []
#     lnslist += ax1.plot(growthData['Time'][growthData['Time'] < nDaysToFit],
#              growthData.FluorescentArea[growthData['Time'] < nDaysToFit],
#              linestyle="None", marker=kwargs.get('markerTypeTrD', "x"), markersize=kwargs.get('markerSizeTrD', 10),
#              color="black", markeredgewidth=2, label="Training Data")
#     # Observed future growth
#     lnslist += ax1.plot(growthData['Time'][growthData['Time'] > nDaysToFit],
#              growthData.FluorescentArea[growthData['Time'] > nDaysToFit],
#              linestyle="None", marker=".", color="green", markersize=kwargs.get('markerSizeTeD', 20),
#              label="Testing Data")
#
#     # Plot the drug concentration
#     drugConcentrationVec = TreatmentListToTS(treatmentList=ExtractTreatmentFromDf(growthData),
#                                              tVec=growthData['Time'])
#     drugLn = ax2.fill_between(growthData['Time'],
#                      0, drugConcentrationVec, color="#8f59e0", alpha=0.2, label="Drug Concentration")
#     # Format the plot
#     ax1.set_xlim([0, kwargs.get('xLim', 1.1*growthData['Time'].max())])
#     ax1.set_ylim([kwargs.get('yMin',0), kwargs.get('yLim', 1.1*growthData.FluorescentArea.max())])
#     ax2.set_xlim([0, kwargs.get('xLim', 1.1 * growthData['Time'].max())])
#     ax2.set_ylim([0, kwargs.get('y2lim', max(1.1*growthData.DrugConcentration.max(),.1))])
#     ax1.set_ylabel("Fluorescent Area")
#     ax1.set_xlabel("Days since start of the experiment")
#     ax2.set_ylabel(r"Drug Concentration in $\mu M$")
#     ax1.set_title(kwargs.get('titleStr',''))
#     if nDaysToFit<growthData['Time'].max(): ax1.vlines(x=nDaysToFit, ymin=0, ymax=5e5, linestyle="--")
#     if kwargs.get('plotLegendB', True):
#         labsList = [l.get_label() for l in lnslist]
#         ax2.legend(lnslist, labsList, loc=kwargs.get('legendLoc', "upper left"))
#     fig.tight_layout()
#     if kwargs.get('saveFigB', False):
#         ax1.savefig(kwargs.get('outName', 'growthData_%s.png'%spheroidId), orientation='portrait', format='png')
#     if kwargs.get('returnAx', False): return (ax1, ax2)

# Plot the 96 well plate
def PlotData_asPanel(dataDf, feature='FluorescentArea', colList=None, **kwargs):
    # Parameterise
    yLim = kwargs.get('yLim', dataDf[feature].max())
    featureStr = kwargs.get('featureStr', feature)
    if colList is None: colList = np.arange(1,13)
    nCols = len(colList)

    plt.figure(figsize=kwargs.get('figSize',(25,20)))
    for colIdx,colNum in tqdm(enumerate(colList), disable=(not kwargs.get('progressBarB', True))):
        for rowIdInt,rowId in enumerate(list(string.ascii_uppercase[:8])):
            spheroidId = rowId+str(colNum)
            pltId = np.ravel_multi_index([[rowIdInt],[colIdx]],(8,nCols))+1
            ax1 = plt.subplot(8,nCols,pltId)
            # Extract data for current spheroid
            dataToPlot = dataDf[dataDf.SpheroidId == spheroidId]
            if dataToPlot.shape[0]<1:
                continue
            # Generate Plot
            kwargs['yLim'] = yLim
            ax1,ax2 = PlotData(dataToPlot, titleStr=spheroidId, plotLegendB=False, returnAx=True, **kwargs)
            # Remove y-labels everywhere except for left margin
            if colIdx > 0: # Turn off left y-axis
                ax1.yaxis.set_ticklabels([])
                ax1.yaxis.set_ticks([])
                ax1.set_ylabel("")
            if colIdx < nCols-1: # Turn off right y-axis
                ax2.yaxis.set_ticklabels([])
                ax2.yaxis.set_ticks([])
                ax2.set_ylabel("")
            # Remove x-labels everywhere except on bottom
            if rowId < "H":
                ax1.xaxis.set_ticklabels([])
                ax1.set_xlabel("")
            else:
                ax1.set_xlabel("Days Since Start")
            # If requested add the treatment names in the titles of the top row
            if kwargs.get('titleRowB',False):
                if rowId == "A":
                    currBranchNameStr = dataToPlot.TreatmentBranch.unique()[0]
                    plt.title(currBranchNameStr + "\n" + spheroidId)
    plt.tight_layout()
    if kwargs.get('saveFigB', False):
        plt.savefig(kwargs.get('outName', 'growthData_panel.png'), orientation='portrait', format='png')


# ====================================================================================
# Functions for dealing with treatment schedules
# ====================================================================================
# Helper function to extract the treatment schedule from the data
def ConvertTDToTSFormat(timeVec,drugIntensityVec):
    treatmentScheduleList = [] # Time intervals in which we have the same amount of drug
    tStart = timeVec[0]
    currDrugIntensity = drugIntensityVec[0]
    for i,t in enumerate(timeVec):
        if drugIntensityVec[i]!=currDrugIntensity and not (np.all(np.isnan([drugIntensityVec[i],currDrugIntensity]))): # Check if amount of drug has changed
            treatmentScheduleList.append([tStart,t,currDrugIntensity])
            tStart = t
            currDrugIntensity = drugIntensityVec[i]
    treatmentScheduleList.append([tStart,timeVec[-1]+(tStart==timeVec[-1])*1,currDrugIntensity])
    return treatmentScheduleList

# Helper function to obtain treatment schedule from calibration data
def ExtractTreatmentFromDf(dataDf):
    timeVec = dataDf['Time'].values
    nDaysPreTreatment = int(timeVec.min())
    if nDaysPreTreatment != 0: # Add the pretreatment phase if it's not already added
        timeVec = np.concatenate((np.arange(0, nDaysPreTreatment), timeVec), axis=0)
    drugIntensityVec = dataDf.DrugConcentration.values
    drugIntensityVec = np.concatenate((np.zeros((nDaysPreTreatment,)), drugIntensityVec), axis=0)
    return ConvertTDToTSFormat(timeVec, drugIntensityVec)

# Turns a treatment schedule in list format (i.e. [tStart, tEnd, DrugConcentration]) into a time series
def TreatmentListToTS(treatmentList,tVec):
    drugConcentrationVec = np.zeros_like(tVec)
    for drugInterval in treatmentList:
        drugConcentrationVec[(tVec>=drugInterval[0]) & (tVec<=drugInterval[1])] = drugInterval[2]
    return drugConcentrationVec

# Extract the date as a datetime object from a model or experiment data frame
def GetDateFromDataFrame(df):
    year, month, day, hour, minute = [df[key].values[0] for key in ['Year','Month','Day','Hour','Minute']]
    hour = 12 if np.isnan(hour) else hour
    minute = 0 if np.isnan(minute) else minute
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))


# ====================================================================================
# Misc
# ====================================================================================
def printTable(myDict, colList=None, printHeaderB=True, colSize=None, **kwargs):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList: colList = list(myDict[0].keys() if myDict else [])
    myList = [colList] if printHeaderB else [] # 1st row = header
    for item in myDict: myList.append([str('%.2e'%item[col] or '') for col in colList])
    colSize = [max(map(len,col)) for col in zip(*myList)] if not colSize else colSize
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    if printHeaderB: myList.insert(1, ['-' * i for i in colSize]) # Seperating line
    for item in myList: print(formatStr.format(*item))
    if kwargs.get('getColSizeB',False): return colSize

def mkdir(dirName):
    """
    Recursively generate a directory or list of directories. If directory already exists be silent. This is to replace
    the annyoing and cumbersome os.path.mkdir() which can't generate paths recursively and throws errors if paths
    already exist.
    :param dirName: if string: name of dir to be created; if list: list of names of dirs to be created
    :return: Boolean
    """
    dirToCreateList = [dirName] if type(dirName) is str else dirName
    for directory in dirToCreateList:
        currDir = ""
        for subdirectory in directory.split("/"):
            currDir = os.path.join(currDir, subdirectory)
            try:
                os.mkdir(currDir)
            except:
                pass
        return True