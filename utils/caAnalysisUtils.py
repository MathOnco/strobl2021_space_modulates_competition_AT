# ====================================================================================
# Library with functions used for the analysis of the on-lattice adaptive therapy CA model.
# ====================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append("./utils")
from odeAnalysisUtils import Simulate_AT_FixedThreshold, rdModel_nsKill

# -------------------------------- RunSimulation() ----------------------------------
# Python wrapper around the jar file for the CA.
# --------------------------------------------------------------------------------------
def RunSimulation(paramDic,jarFileName="onLatticeModel.jar",printCommand=False):
    argStr = " "
    for var in paramDic.keys():
        argStr += "-%s %s "%(var,paramDic[var])
    if printCommand: print("java -jar %s"%jarFileName+argStr)
    os.system("java -jar %s"%jarFileName+argStr)

# -------------------------------- PlotSimulationCA() ----------------------------------
# Plot a set of simulations with mean and confidence intervals
# --------------------------------------------------------------------------------------
def PlotSimulation_CA(dataDf, plotSizeB=True, plotPopsB=True, aggregateData=True,
                      yLimVec=[0,1.5e4], y2lim=1, decoratex=True, decoratey=True, decoratey2=False,
                      showProgression=False, minTime=150, drugBarPosition=0.85, drugBarColour = 'black',
                      secondaryDrugPalette = {"AT+PPTx":"#1DB100","AT+PTTx":"#FF95CA"},
                      titleStr="", ax=None, figsize=(10,8), savefigB=False, **kwargs):
    # Set up the figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    lnslist = []

    # Plot the size the we will see on the images
    if plotSizeB:
        sns.lineplot(x="Time",y="V", hue="TreatmentBranch", style="TreatmentBranch", ci='sd',
                     lw=kwargs.get('linewidthA', 7),
                     palette=kwargs.get('colorA', ['b']),
                     estimator= 'mean' if aggregateData else None,
                     data=dataDf,ax=ax)

    # Plot the individual populations
    if plotPopsB:
        sns.lineplot(x="Time",y="S",hue="TreatmentBranch", style="TreatmentBranch", ci='sd',
                     lw=kwargs.get('linewidth', 7), palette=[kwargs.get('colorS', "#0F4C13")],
                     estimator= 'mean' if aggregateData else None,
                     data=dataDf,ax=ax)
        ax.lines[2].set_linestyle(kwargs.get('linestyleS', '--'))
        sns.lineplot(x="Time",y="R",hue="TreatmentBranch", style="TreatmentBranch", ci='sd',
                     lw=kwargs.get('linewidth', 7), palette=[kwargs.get('colorR', '#710303')],
                     estimator= 'mean' if aggregateData else None,
                     data=dataDf,ax=ax)
        ax.lines[4].set_linestyle(kwargs.get('linestyleR', '-.'))


    # Plot the drug concentration
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    if len(dataDf.ReplicateId.unique())>1:
        exampleReplicateId = dataDf.ReplicateId.unique()[0]
        timeVec = dataDf.Time[dataDf.ReplicateId==exampleReplicateId]
        drugConcentrationVec = dataDf.DrugConcentration[dataDf.ReplicateId==exampleReplicateId]
    else:
        timeVec = dataDf.Time
        drugConcentrationVec = dataDf.DrugConcentration
    drugConcentrationVec = drugConcentrationVec/(1-drugBarPosition)+drugBarPosition
    ax2.fill_between(timeVec,
                     drugBarPosition, drugConcentrationVec, color=drugBarColour, step='post',
                     alpha=1., label="Drug Concentration")
    txName = dataDf.TreatmentBranch.unique()[0]
    if txName in ["AT+PPTx","AT+PTTx"]:
        valVec = np.unique(drugConcentrationVec)
        drugConcentrationVec[drugConcentrationVec == valVec[0]] = -1
        drugConcentrationVec[drugConcentrationVec == valVec[1]] = valVec[0]
        drugConcentrationVec[drugConcentrationVec == -1] = valVec[1]
        ax2.fill_between(timeVec,
                         drugBarPosition, drugConcentrationVec, color=secondaryDrugPalette[txName],
                         step='post', alpha=1., label="Drug Concentration")

    # Add lines to indicate the initial size and the size and time at progression
    ax.hlines(xmin=0,xmax=kwargs.get('xlim', 1.1*dataDf.Time.max()),
              y=dataDf.V.iloc[0],linestyles=':',linewidth=6,color="black")
    if showProgression:
        ax.hlines(xmin=0, xmax=kwargs.get('xlim', 1.1 * dataDf.Time.max()),
                  y=1.2*dataDf.V.iloc[0], linestyles=':', linewidth=6)
        # Compute the TTP
        ttpArr = np.array([dataDf[(dataDf.ReplicateId == i) & (dataDf.V > 1.2*dataDf.V.iloc[0]) & (
                dataDf.Time > minTime)].Time.min() for i in dataDf.ReplicateId.unique()])
        ttpArr = ttpArr[np.isnan(ttpArr) == False]
        meanTTP = np.mean(ttpArr)
        ax.vlines(x=meanTTP, ymin=0, ymax=1.3e4, colors=kwargs.get('colorA', ['b']), linestyles='--', linewidth=6)
        if len(ttpArr) > 1:
            ax.fill_between([np.percentile(ttpArr, 25), np.percentile(ttpArr, 75)], 0, 1.3e4,
                            color=kwargs.get('colorA', ['b']), alpha=.2)

    # Format the plot
    ax.set_xlim([0, kwargs.get('xlim', 1.1*dataDf.Time.max())])
    ax.set_ylim(yLimVec)
    ax2.set_ylim([0, y2lim])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=24)
    if not decoratex:
        ax.set_xticklabels("")
    if not decoratey:
        ax.set_yticklabels("")
    if decoratey2:
        ax2.set_ylabel(r"Drug Dose")
    else:
        ax2.axis("off")
    plt.title(titleStr)
    if kwargs.get('plotLegendB', False):
        labsList = [l.get_label() for l in lnslist]
        plt.legend(lnslist, labsList, loc=kwargs.get('legendLoc', "upper right"))
    else:
        ax.legend().remove()
    plt.tight_layout()
    if savefigB:
        plt.savefig(kwargs.get('outName', 'modelPrediction.png'))

# ------------------------- GenerateATComparisonPlot_CA() ------------------------------
# Plot the AT trajectory from a CA simulation with markers for the TTP under CT and AT.
# As it is computing the TTP of CT, the data also needs to contain the CT data.
# --------------------------------------------------------------------------------------
def GenerateATComparisonPlot_CA(dataDf, t_end=1000, minTime=0, decoratex=True, decoratey=True,
                                aggregateData=True, cmap={"MTD":'#FF9409',"AT50":'#094486'},
                                treatmentToPlot="AT", plotDrugForCT=False, drugBarPalette={"AT50":'black'},
                                drugBarPosition=0.85,
                                xlim=None, ylim=None, y2lim=1, showProgression=False, printTTPB=False, atName="AT50",
                                savefigB=False, outName='caSimulation.png', figsize=(8, 5), ax=None):
    if ax is None: fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
    currDataDf = dataDf[dataDf.TreatmentBranch == atName]
    if treatmentToPlot == "AT":
        PlotSimulation_CA(currDataDf, xlim=t_end, colorA=[cmap.get(atName,'#094486')], plotPopsB=True, yLimVec=[0, ylim],
                          showProgression=showProgression, aggregateData=aggregateData, ax=ax)
    if treatmentToPlot == "Both":
        PlotSimulation_CA(dataDf[dataDf.TreatmentBranch == "MTD"], drugBarColour="white", xlim=t_end,
                          colorA=[cmap["MTD"]], plotPopsB=False, yLimVec=[0, ylim], showProgression=False,
                          aggregateData=aggregateData, ax=ax)
        PlotSimulation_CA(currDataDf, xlim=t_end, colorA=[cmap.get(atName,'#094486')], plotPopsB=False, yLimVec=[0, ylim],
                          showProgression=showProgression, aggregateData=aggregateData, ax=ax)

    initialTumourSize = currDataDf.V.iloc[0].min()
    # Plot the drug bars
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    if plotDrugForCT:
        barWidth = (1 - drugBarPosition)
        for i, txName in enumerate(dataDf.TreatmentBranch.unique()):
            if len(dataDf.ReplicateId.unique()) > 1: # If there are multiple replicates, pick the first replicate as an example of the treatment schedule observed.
                currDataDf = dataDf[dataDf.TreatmentBranch == txName]
                exampleReplicateId = currDataDf.ReplicateId.unique()[0]
                timeVec = currDataDf.Time[currDataDf.ReplicateId == exampleReplicateId]
                drugConcentrationVec = currDataDf.DrugConcentration[currDataDf.ReplicateId == exampleReplicateId]
            else:
                timeVec = dataDf.Time
                drugConcentrationVec = dataDf.DrugConcentration
            drugConcentrationVec = drugConcentrationVec * barWidth + 1 - (i + 1) * barWidth
            ax2.fill_between(timeVec,
                             1 - (i + 1) * barWidth, drugConcentrationVec, color=drugBarPalette[txName],
                             step='post',
                             alpha=1., label="Drug Concentration")
    # Obtain  and mark the TTPs
    for txName in ["MTD", atName]:
        currDataDf = dataDf[dataDf.TreatmentBranch == txName]
        color = [(1, 148 / 255, 9 / 255)] if txName == "MTD" else [(9 / 255, 68 / 255, 134 / 255)]
        ls = '-' if txName == "MTD" else '--'
        ttpArr = np.array([currDataDf[(currDataDf.ReplicateId == i) & (currDataDf.V > 1.2 * initialTumourSize) & (
                    currDataDf.Time > minTime)].Time.min() for i in currDataDf.ReplicateId.unique()])
        ttpArr = ttpArr[np.isnan(ttpArr) == False]
        meanTTP = np.mean(ttpArr)
        ax.vlines(x=meanTTP, ymin=0, ymax=1.3e4, colors=color, linestyles=ls, linewidth=6)
        if len(ttpArr) > 1:
            ax.fill_between([np.percentile(ttpArr, 25), np.percentile(ttpArr, 75)], 0, 1.3e4,
                            color=color, alpha=.2)
        if printTTPB:
            print("%s Mean TTP: %1.2f" % (txName, meanTTP))
    if xlim is not None: ax.set_xlim(xlim)
    ax2.set_ylim([0, y2lim])
    ax2.axis("off")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not decoratex:
        ax.set_xticklabels("")
    if not decoratey:
        ax.set_yticklabels("")
    if savefigB:
        plt.savefig(outName)

# ------------------------- GenerateATComparisonPlot_ODE() ------------------------------
# Show the AT trajectory for a parameter set for the ODE model. Uses the same plotting functions
# as the above CA functions, so is ideal for comparing the ODE and the CA model.
# --------------------------------------------------------------------------------------
def GenerateATComparisonPlot_ODE(initialTumourSize, sFrac, paramDic=None,
                                 t_end=1500, relToPopEq=False,
                                 decorateX=True, decorateY=True,
                                 intervalLength=.5, nTimePts=100.,
                                 xlim=None, ylim=.3 * 1e4, figsize=(8, 5),
                                 outName=None, ax=None):
    if relToPopEq: initialTumourSize *= (1 - paramDic['dS'] / paramDic['rS'])
    initialStateVec = [initialTumourSize * sFrac, initialTumourSize * (1 - sFrac), 0, paramDic['DMax']]
    initialStateVec[2] = paramDic['theta'] * (initialStateVec[0] + initialStateVec[1])

    # Simulate
    tmpDfList = []
    resultsDf = Simulate_AT_FixedThreshold(modelFun=rdModel_nsKill,
                                           initialStateVec=initialStateVec,
                                           atThreshold=1., intervalLength=t_end,
                                           paramDic=paramDic,
                                           t_end=t_end, t_eval=np.linspace(0, t_end, nTimePts))
    resultsDf['TreatmentBranch'] = 'MTD'
    tmpDfList.append(resultsDf)
    resultsDf = Simulate_AT_FixedThreshold(modelFun=rdModel_nsKill,
                                           initialStateVec=initialStateVec,
                                           atThreshold=.5, intervalLength=intervalLength,
                                           paramDic=paramDic,
                                           t_end=t_end, t_eval=np.linspace(0, t_end, nTimePts))
    resultsDf['TreatmentBranch'] = 'AT50'
    tmpDfList.append(resultsDf)

    # Plot the results
    dataDf = pd.concat(tmpDfList)
    dataDf['ReplicateId'] = 0
    dataDf['S'] *= 1e4
    dataDf['R'] *= 1e4
    dataDf['V'] *= 1e4
    dataDf.rename(columns={'D': 'DrugConcentration'}, inplace=True)
    GenerateATComparisonPlot_CA(dataDf, aggregateData=False,
                                xlim=xlim, ylim=ylim, t_end=xlim[1],
                                decoratex=decorateX, decoratey=decorateY,
                                ax=ax, figsize=figsize, savefigB=outName != None,
                                outName=outName)

# ------------------------- PlotCompetitionOverTime() ------------------------------
# Function to plot the strength of competition over time (measured as the % of blocked divisions).
# --------------------------------------------------------------------------------------
def PlotCompetitionOverTime(dataDf, competitionMetric="PropFailedDivs",
                            errStyle=0.95,
                            drugBarPosition=0.85, plotDrugForCT=False,
                            showProgression=False, minTime=150,
                            xlim=None, ylim=None, y2lim=1/1.2,
                            decoratex=True, decoratey=True,
                            cmap=[(1, 148 / 255, 9 / 255), (9 / 255, 68 / 255, 134 / 255)],
                            figsize=(12, 5), ax=None):
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(x="Time", y=competitionMetric, ci=errStyle,
                 hue="TreatmentBranch", style="TreatmentBranch",
                 palette=cmap,
                 lw=7,
                 ax=ax, data=dataDf)
    # Plot vertical bars to indicate progression
    if showProgression:
        for k, txName in enumerate(dataDf.TreatmentBranch.unique()):
            # Compute the TTP
            ttpArr = np.array([dataDf[((dataDf.TreatmentBranch == txName)) & (dataDf.ReplicateId == i) &
                                      (dataDf.NCells > 1.2 * dataDf.NCells.iloc[0]) &
                                      (dataDf.Time > minTime)].Time.min() for i in dataDf.ReplicateId.unique()])
            ttpArr = ttpArr[np.isnan(ttpArr) == False]
            meanTTP = np.mean(ttpArr)
            ax.vlines(x=meanTTP, ymin=0, ymax=1.3e4, colors=cmap[k],
                      linestyles='--', linewidth=6)
            if len(ttpArr) > 1:
                ax.fill_between([np.percentile(ttpArr, 25), np.percentile(ttpArr, 75)], 0, 1.3e4,
                                color=cmap[k], alpha=.2)

    # Plot the drug bars
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    barWidth = (1 - drugBarPosition)
    for i, txName in enumerate(dataDf.TreatmentBranch.unique()):
        if plotDrugForCT == False and txName == "MTD": continue
        if len(dataDf.ReplicateId.unique()) > 1:
            currDataDf = dataDf[dataDf.TreatmentBranch == txName]
            exampleReplicateId = currDataDf.ReplicateId.unique()[0]
            timeVec = currDataDf.Time[currDataDf.ReplicateId == exampleReplicateId]
            drugConcentrationVec = currDataDf.DrugConcentration[currDataDf.ReplicateId == exampleReplicateId]
        else:
            timeVec = dataDf.Time
            drugConcentrationVec = dataDf.DrugConcentration
        drugConcentrationVec = drugConcentrationVec * barWidth + 1 - (i + 1) * barWidth
        ax2.fill_between(timeVec,
                         1 - (i + 1) * barWidth, drugConcentrationVec, color="black",
                         step='post',
                         alpha=1., label="Drug Concentration")
        if txName == "AT+PPTx":
            valVec = np.unique(drugConcentrationVec)
            drugConcentrationVec[drugConcentrationVec == valVec[0]] = -1
            drugConcentrationVec[drugConcentrationVec == valVec[1]] = valVec[0]
            drugConcentrationVec[drugConcentrationVec == -1] = valVec[1]
            ax2.fill_between(timeVec,
                             1 - (i + 1) * barWidth, drugConcentrationVec, color="#1DB100",
                             step='post',
                             alpha=1., label="Drug Concentration")
        if txName == "AT+PTTx":
            valVec = np.unique(drugConcentrationVec)
            drugConcentrationVec[drugConcentrationVec == valVec[0]] = -1
            drugConcentrationVec[drugConcentrationVec == valVec[1]] = valVec[0]
            drugConcentrationVec[drugConcentrationVec == -1] = valVec[1]
            ax2.fill_between(timeVec,
                             1 - (i + 1) * barWidth, drugConcentrationVec, color="#FF95CA",
                             step='post',
                             alpha=1., label="Drug Concentration")

    if xlim is not None: ax.set_xlim([0, xlim])
    if ylim is not None: ax.set_ylim([0, ylim])
    ax2.set_ylim([0, y2lim])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax2.axis("off")
    if not decoratex:
        ax.set_xticklabels("")
    if not decoratey:
        ax.set_yticklabels("")
    ax.tick_params(labelsize=32)
    ax.get_legend().remove()
    sns.despine(offset=5, trim=True)
    plt.xticks(rotation=45);

# ------------------------- PlotTTPHeatmap() ------------------------------
# Function to plot a heatmap of the ttp for a parameter sweep.
# --------------------------------------------------------------------------------------
def PlotTTPHeatmap(timeGainedMat, feature='RelTimeGained', cmap="Greys", vmin=0, vmax=45, cbar=True, annot=True,
                   fmt="1.0f", ax=None):
    if ax is None: _, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(timeGainedMat,
                vmin=vmin, vmax=vmax,
                cmap=cmap, linewidths=2,
                annot=annot, fmt=fmt,
                cbar=cbar, ax=ax)
    ax.tick_params(labelsize=24, rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("")

# ------------------------- QQPlot() ------------------------------
# Function to compare ODE and CA model using qq plots
# --------------------------------------------------------------------------------------
def QQPlot(dataDf,x="TTP_AT50_ODE",y="TTP_AT50_CA",groupingVariables=["RFrac","InitialTumourSize"],ci='sd',
           axLimVec=None,palette=None,sizeDic=None,ax=None):
    if ax==None: fig,ax = plt.subplots(1,1,figsize=(7,7))
    # Plot the lines
    sns.lineplot(x=x,y=y,err_style='band',ci=ci,
                 hue=groupingVariables[0], style=groupingVariables[0],
                 markers=True,markersize=1,
                 palette=palette,
                 ax=ax,data=dataDf)
    # Add the markers in different sizes to reflect n0
    summaryDf = dataDf.groupby(groupingVariables).mean()
    summaryDf.reset_index(inplace=True)
    sns.scatterplot(x=x,y=y, style=groupingVariables[0],
                    hue=groupingVariables[0], palette=palette,
                    size=groupingVariables[1],
                    markers=True, sizes=sizeDic,
                    ax=ax,
                    legend=False, data=summaryDf)
    # Set x and y axis to equal scale
    if axLimVec is None:
        xLimVec = ax.get_xlim()
        yLimVec = ax.get_ylim()
        axLimVec = [min(xLimVec[0],yLimVec[0]),max(xLimVec[1],yLimVec[1])]
    ax.set_xlim(axLimVec)
    ax.set_ylim(axLimVec)

    # Plot diagonal line to show where TTP would be equal
    xVec = np.linspace(*axLimVec,100)*0.9
    ax.plot(xVec,xVec,color=sns.xkcd_rgb['blue grey'],lw=3,ls='--')

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=24)
    ax.get_legend().remove()
    sns.despine(ax=ax,offset=10, trim=False)