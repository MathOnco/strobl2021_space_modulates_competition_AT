import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import sympy as sm
from tqdm import tqdm

# Format plot
sns.set(style="white",
        font_scale=1.5,
        rc={'figure.figsize':(12,6)})

# --------------------------- Model Equations -----------------------------------------------------
def basicModel_logKill(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = (1 - s - paramDic['cRS']*r)*s - paramDic['d']*c*s
    dudtVec[1] = paramDic['p']*(1 - paramDic['cSR']*s - r)*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def basicModel_diffK(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = (1 - s - r)*s - paramDic['d']*c*s
    dudtVec[1] = paramDic['p']*(1 - paramDic['cSR']*(s + r))*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + paramDic['cSR']*dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def basicModel_divKill(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = (1 - s - r)*s*(1 - paramDic['d']*c)
    dudtVec[1] = paramDic['p']*(1 - s - r)*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def nortonSimon_diffK(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] =  (1 - s - r)*s*(1 - paramDic['d']*c)
    dudtVec[1] = paramDic['p']*(1 - paramDic['cSR']*(s + r))*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + paramDic['cSR']*dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def basicModel_switching(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = (1 - s - paramDic['cRS']*r)*s - paramDic['a']*c*s + paramDic['b']*(1-c)*r - paramDic['d']*c*s
    dudtVec[1] = paramDic['p']*(1 - paramDic['cSR']*s - r)*r + paramDic['a']*c*s - paramDic['b']*(1-c)*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def rhModel_logKill(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = (1 - s - paramDic['cRS'] * r) * s - paramDic['d']*c*s
    dudtVec[1] = (paramDic['p'] - paramDic['k'] * (r + paramDic['cSR'] * s))*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

def rdModel_nsKill(t, uVec, paramDic):
    s, r, v, c = uVec
    dudtVec = np.zeros_like(uVec)
    dudtVec[0] = paramDic['rS']*(1 - s - paramDic['cRS'] * r) * (1-paramDic['dD']*c) * s - paramDic['dS']*s
    dudtVec[1] = paramDic['rR']*(1 - (r + paramDic['cSR'] * s)/paramDic['k'])*r - paramDic['dR']*r
    dudtVec[2] = paramDic['theta']*(dudtVec[0] + dudtVec[1])
    dudtVec[3] = 0
    return (dudtVec)

# --------------------------- Model Simulation -----------------------------------------------------
def Simulate_ContinousTx(initialStateVec,paramDic,modelFun,t_end=None,t_span=None,t_eval=None,nTimePts=100,**kwargs):
    t_span = t_span if t_span is not None else (0, t_end)
    t_eval = t_eval if t_eval is not None else np.linspace(t_span[0],t_span[1],nTimePts)
    solObj = scipy.integrate.solve_ivp(lambda t, uVec: modelFun(t,uVec,paramDic), y0=initialStateVec,
                                       t_span=t_span,t_eval=t_eval,**kwargs)
    return pd.DataFrame({"Time": solObj.t, "S": solObj.y[0, :], "R": solObj.y[1, :],
                              "V":solObj.y[2,:], "D": solObj.y[3, :]})

def Simulate_AT_FixedThreshold(initialStateVec,paramDic,modelFun,
                               atThreshold=0.5,intervalLength=3,refSize=None,t_end=None,t_span=None,t_eval=None,nTimePts=100,**kwargs):
    t_span = t_span if t_span is not None else (0, t_end)
    t_eval = t_eval if t_eval is not None else np.linspace(0,t_end,nTimePts)
    resultsDFList = []
    currInterval = [t_span[0],t_span[0]+intervalLength]
    refSize = initialStateVec[2] if refSize is None else refSize
    dose = initialStateVec[-1]
    currCycleId = 0
    while currInterval[1] <= t_end:
        # Simulate
        resultsDf = Simulate_ContinousTx(initialStateVec,modelFun=modelFun,
                                         paramDic=paramDic,
                                         t_span=(currInterval[0], currInterval[1]),
                                         t_eval=np.linspace(currInterval[0], currInterval[1],1000))
        resultsDf['CycleId'] = currCycleId
        resultsDFList.append(resultsDf)

        # Update dose
        if resultsDf.V.iat[-1] > refSize:
            currCycleId += (dose==0)
            dose = paramDic['DMax']
        elif resultsDf.V.iat[-1] < (1-atThreshold)*refSize:
            dose = 0
        else:
            dose = (dose > 0)*paramDic['DMax']
        initialStateVec = [resultsDf.S.iat[-1], resultsDf.R.iat[-1], resultsDf.V.iat[-1], dose]

        # Update interval
        currInterval = [x+intervalLength for x in currInterval]
    resultsDf = pd.concat(resultsDFList)
    # Interpolate to the desired time grid
    trimmedResultsDic = {'Time':t_eval}
    for variable in ['S','R','V','D','CycleId']:
        f =  scipy.interpolate.interp1d(resultsDf.Time,resultsDf[variable],fill_value="extrapolate")
        trimmedResultsDic = {**trimmedResultsDic,variable:f(t_eval)}
    return pd.DataFrame(data=trimmedResultsDic)

# --------------------------- Analysis -----------------------------------------------------
def ProfileTreatmentStrategies(modelFun,paramDic,atThresholdList=[0.3, 0.5],intervalLength=0.3,dt=1.,
                               initialSizeList=np.linspace(0.25,0.75,5),
                               rFracList=[0.1,0.01,0.001],
                               tumourSizeWhenProgressed=1.2,cureThreshold=0.1,enableProgressBar=True):
    tmpDicList = []
    for initialTumourSize,rFrac in tqdm([(x,y) for x in initialSizeList for y in rFracList],disable=enableProgressBar==False):
        initialStateVec,_ = GenerateParameterDic(initialSize=initialTumourSize,rFrac=rFrac,cost=0,turnover=0,paramDic=paramDic)
        maxTolerableBurden = initialTumourSize*tumourSizeWhenProgressed
        tumourBurdenWhenCured = cureThreshold*initialTumourSize

        # 0. Check that tumour can progress
        rMax = np.inf
        if modelFun.__name__=='rdModel_nsKill':
            rMax = (1-paramDic['dR']/paramDic['rR'])*paramDic['k']
        if modelFun.__name__=='odeModel_densDepDeath' and paramDic['dR']>0:
            rMax = paramDic['rR']/paramDic['dR']*paramDic['K']
        if rMax < maxTolerableBurden:
            for strategy in ["MTD"]+[str(thresh) for thresh in atThresholdList]:
                tmpDicList.append({"TreatmentBranch": strategy,
                                   "RFrac": rFrac, "InitialTumourSize": initialTumourSize,
                                   "Cured": False, "IndefiniteControl":True, "TimeToFailure": np.nan, "RelTimeGained": np.nan,
                                   "NCycles": np.nan})
            continue

        # 1. Simulate MTD until cure or failure
        r0 = max(initialStateVec[1],0.001) # Make sure I assume at least some r0 for this calculation, otherwise it blows up
        rR = paramDic.get('rR',paramDic.get('p',0.03))
        t_end = np.log((1-r0)/(r0*(1/maxTolerableBurden-1)))/rR
        finalTumourBurden = 0
        cured = False
        currStateVec = initialStateVec
        currTime = 0
        # a) Figure out how long to simulate for. If MTD cures the tumour we skip this
        # scenario. If it doesn't, then we will simulate until 2x TTP of MTD. To do so,
        # we here have to determine TTP.
        while finalTumourBurden<maxTolerableBurden-1e-6 and not cured: # 1e-6 to deal with cases when the maxBurden is so close to KEff that the ODE solver's numerical error becomes a problem.
            resultsDf = Simulate_ContinousTx(initialStateVec=currStateVec,
                                             modelFun=modelFun,paramDic=paramDic,
                                             t_span=(currTime,t_end))
            finalTumourBurden = resultsDf.V.iloc[-1]
            cured = np.any(resultsDf.V<tumourBurdenWhenCured)
            currStateVec = [resultsDf.S.iloc[-1], resultsDf.R.iloc[-1], resultsDf.V.iloc[-1], 1]
            currTime = t_end
            t_end *= 1.1

        # Don't simulate if MTD cures the tumour
        if cured:
            tmpDicList.append({"TreatmentBranch":"MTD",
                               "RFrac": rFrac, "InitialTumourSize":initialTumourSize,
                               "Cured":True, "IndefiniteControl":False, "TimeToFailure":np.nan,"RelTimeGained":np.nan,
                               "NCycles":np.nan})
            continue

        # b) Get the estimate for TTP using the same function and time grid that I will be using for the AT.
        # Do this so that both the simulation method, and comparison grid are identical to that used for AT.
        resultsDf = Simulate_AT_FixedThreshold(initialStateVec,modelFun=modelFun,
                                               paramDic=paramDic,
                                               atThreshold=1.,
                                               refSize=initialTumourSize,
                                               intervalLength=intervalLength,
                                               t_eval=np.arange(0,t_end+dt,dt),
                                               t_end=t_end)
        timeToFailure_MTD = resultsDf.Time[resultsDf.V>maxTolerableBurden-1e-6].min() # np.abs to deal with cases when we are so close to the threshold that numerical error can cause problems
        tmpDicList.append({"TreatmentBranch":"MTD", "RFrac": rFrac, "InitialTumourSize":initialTumourSize,
                            "Cured":False, "IndefiniteControl":False, "TimeToFailure":timeToFailure_MTD,"RelTimeGained":np.nan,
                           "NCycles":np.nan})

        # 2. Simulate AT
        for atThreshold in atThresholdList:
            t_end = 2*timeToFailure_MTD
            timeToFailure = np.nan
            while np.isnan(timeToFailure):
                resultsDf = Simulate_AT_FixedThreshold(initialStateVec,modelFun=modelFun,
                                                       paramDic=paramDic,
                                                       atThreshold=atThreshold,
                                                       intervalLength=intervalLength,
                                                       t_eval=np.arange(0,t_end+dt,dt),
                                                       t_end=t_end)
                timeToFailure = resultsDf.Time[resultsDf.V>maxTolerableBurden-1e-6].min()
                t_end *= 1.1
            # Assess Performance
            relTimeGained = (timeToFailure-timeToFailure_MTD)/timeToFailure_MTD*100
            tmpDicList.append({"TreatmentBranch":str(atThreshold),
                               "RFrac": rFrac, "InitialTumourSize":initialTumourSize,
                               "Cured":False, "IndefiniteControl":False, "TimeToFailure":timeToFailure,
                               "AbsTimeGained":timeToFailure-timeToFailure_MTD,"RelTimeGained":relTimeGained,
                               "NCycles":resultsDf.CycleId.max()})
    return pd.DataFrame(tmpDicList)

# --------------------------- Plotting Functions -----------------------------------------------------
def PlotSimulation(dataDf,drugConcentrationVec=None,sizeVar="V",plotSizeB=True,plotPopsB=True,
                       drugBarPosition=0.85,plotLegendB=False,plotDrugBarB=True,
                       yLimVec=[0,1.5], y2lim=1, ax=None, figsize=(10,8), titleStr="", **kwargs):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    lnslist = []
    # Plot the size the we will see on the images
    if plotSizeB:
        lnslist += ax.plot(dataDf.Time,
                            dataDf[sizeVar],
                            lw=kwargs.get('linewidthA', 7), color=kwargs.get('colorA', 'b'),
                            linestyle=kwargs.get('linestyleA', '-'), marker=kwargs.get('markerA', None),
                            label=kwargs.get('labelA', 'Model Prediction'))

    # Plot the individual populations
    if plotPopsB:
        propS = dataDf.S.values / (dataDf.S.values + dataDf.R.values)
        lnslist += ax.plot(dataDf.Time,
                            propS * dataDf[sizeVar],
                            lw=kwargs.get('linewidth', 7), linestyle=kwargs.get('linestyleS', '--'),
                            color=kwargs.get('colorS', "#0F4C13"),
                            label='S')
        lnslist += ax.plot(dataDf.Time,
                            (1 - propS) * dataDf[sizeVar],
                            lw=kwargs.get('linewidth', 7), linestyle=kwargs.get('linestyleR', '--'),
                            color=kwargs.get('colorR', '#710303'),
                            label='R')

    # Plot the drug concentration
    if plotDrugBarB:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        drugConcentrationVec = drugConcentrationVec/(1-drugBarPosition)+drugBarPosition
        ax2.fill_between(dataDf.Time,
                         drugBarPosition, drugConcentrationVec, color="black",
                         alpha=1., label="Drug Concentration")
        ax2.set_ylim([0, y2lim])
        ax2.axis("off")
    # Format the plot
    plt.xlim([0, kwargs.get('xlim', 1.1*dataDf.Time.max())])
    ax.set_ylim(yLimVec)
    ax.tick_params(labelsize=24)
    plt.title(titleStr)
    if plotLegendB:
        labsList = [l.get_label() for l in lnslist]
        plt.legend(lnslist, labsList, loc=kwargs.get('legendLoc', "upper right"))
    plt.tight_layout()
    if kwargs.get('savefigB', False):
        plt.savefig(kwargs.get('outName', 'modelPrediction.png'), orientation='portrait', format='png')

def GenerateATComparisonPlot(initialTumourSize, rFrac, paramDic,
                             modelFun = rdModel_nsKill,
                             t_end=1500, relToPopEq=False,
                             decorateX=True, decorateY=True,
                             intervalLength=1., nTimePts=100., atThreshold=0.5,
                             printDifferenceInTTP=False,
                             ylim=1.3, figsize=(8, 5),
                             colorA='#094486',
                             ax=None,
                             outName=None):
    if relToPopEq: initialTumourSize *= (1 - paramDic['dS'] / paramDic['rS'])
    initialStateVec, _ = GenerateParameterDic(initialSize=initialTumourSize, rFrac=rFrac, cost=0, turnover=0,
                                              paramDic=paramDic)

    # Simulate
    resultsDf = Simulate_AT_FixedThreshold(modelFun=modelFun,
                                           initialStateVec=initialStateVec,
                                           atThreshold=1., intervalLength=t_end,
                                           paramDic=paramDic,
                                           t_end=t_end, t_eval=np.linspace(0, t_end, nTimePts))
    ttp_ct = resultsDf.Time[resultsDf.V > 1.2 * initialTumourSize].min()
    resultsDf = Simulate_AT_FixedThreshold(modelFun=modelFun,
                                           initialStateVec=initialStateVec,
                                           atThreshold=atThreshold, intervalLength=intervalLength,
                                           paramDic=paramDic,
                                           t_end=t_end, t_eval=np.linspace(0, t_end, nTimePts))
    ttp_AT = resultsDf.Time[resultsDf.V > 1.2 * initialTumourSize].min()

    # Plot the results
    if ax is None: fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
    PlotSimulation(resultsDf, sizeVar="V", drugConcentrationVec=resultsDf.D, xlim=t_end,
                   linewidthA=7, linewidth=7,
                   yLimVec=[0, 1.2], y2lim=1, plotLegendB=False, decoratey2=False,
                   labelA='Volume', legendLoc='upper left',
                   colorR='#710303', colorS="#0F4C13", linestyleR='-.',
                   colorA=colorA, ax=ax)
    ax.vlines(x=ttp_ct, ymin=0, ymax=1.2, colors=[(1, 148 / 255, 9 / 255)], linestyles='-', linewidth=6)
    ax.vlines(x=ttp_AT, ymin=0, ymax=1.2, colors=[(9 / 255, 68 / 255, 134 / 255)], linestyles='--', linewidth=6)
    ax.hlines(xmin=0, xmax=t_end, y=initialTumourSize, linestyles=':', linewidth=6)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not decorateX:
        ax.set_xticklabels("")
    if not decorateY:
        ax.set_yticklabels("")
    if outName is not None: plt.savefig(outName)
    if printDifferenceInTTP:
        gainInTTP = ttp_AT - ttp_ct
        relGainInTTP = (ttp_AT - ttp_ct) / ttp_ct * 100
        print("TTP_CT: %1.2f; TTP_AT: %1.2f" % (ttp_ct, ttp_AT))
        print("Relative gain: %1.2f%%; Absolute Gain: %1.2fd" % (relGainInTTP, gainInTTP))

def PlotPhasePlane(fX,fY,varList,paramDic,plotTumourVolumeB=True,plotTumourGrowthRateB=False,
                   plotSGrowthRateB=False,plotRGrowthRateB=False,nLevels_overlay=None,
                   plotSteadyStates=True,solveNumerically=False,
                   xlimVec=[-0.01,1.5],ylimVec=[-0.01,1.5],nPoints=100,
                   figsize=(8,8),lw=5,
                   legendLoc='upper right',titleStr="",decorate=True,
                   arrowsize=1.2, density=[0.8, 2.], arrowLineWidth=1., ax=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    sVec = np.linspace(xlimVec[0], xlimVec[1], nPoints)
    rVec = np.linspace(ylimVec[0], ylimVec[1], nPoints)

    # Plot the vector-field
    fX_lambidified = sm.lambdify(varList, fX)
    FlowFun_X = lambda s, r, paramDic: fX_lambidified(
        *tuple([{"s": s, "r": r, **paramDic}[str(var)] for var in varList]))
    fY_lambidified = sm.lambdify(varList, fY)
    FlowFun_Y = lambda s, r, paramDic: fY_lambidified(
        *tuple([{"s": s, "r": r, **paramDic}[str(var)] for var in varList]))
    X, Y = np.meshgrid(sVec, rVec)
    U = FlowFun_X(X, Y, paramDic)
    V = FlowFun_Y(X, Y, paramDic)
    U[X+Y>(1-paramDic['dS']/paramDic['rS'])] = np.nan
    V[X+Y>(1-paramDic['dS']/paramDic['rS'])] = np.nan
    if plotTumourGrowthRateB:  # Colour by dV/dt
        nLevels_overlay = 25 if nLevels_overlay is None else nLevels_overlay
        dVdt = paramDic['theta'] * (U + V)
        norm = MidpointNormalize(midpoint=0)
        ax.contourf(X, Y, dVdt, nLevels_overlay, cmap=ListedColormap(sns.color_palette("BrBG_r", nLevels_overlay)[::-1]), alpha=0.8, norm=norm)
        #         ax.colorbar();
        contours = ax.contour(X, Y, U + V, 4, colors='black');
        ax.clabel(contours, inline=True, fontsize=11)
    if plotTumourVolumeB:  # Colour by s+r
        nLevels_overlay = 15 if nLevels_overlay is None else nLevels_overlay
        vMat = paramDic['theta'] * (X + Y)
        norm = MidpointNormalize(midpoint=0)
        ax.contourf(X, Y, vMat, nLevels_overlay, cmap=ListedColormap(sns.color_palette("YlOrBr", nLevels_overlay)), alpha=0.8)
        #         ax.colorbar();
        contours = ax.contour(X, Y, vMat, 4, colors='black');
        ax.clabel(contours, inline=True, fontsize=11)
    if plotSGrowthRateB:  # Colour by dS/dt
        nLevels_overlay = 50 if nLevels_overlay is None else nLevels_overlay
        vMat = U
        norm = MidpointNormalize(midpoint=0,vmin=-0.01,vmax=0.01)
        ax.contourf(X, Y, vMat, nLevels_overlay, cmap=ListedColormap(sns.color_palette("BrBG_r", nLevels_overlay)[::-1]), alpha=0.8, norm=norm)
#         contours = ax.contour(X, Y, vMat, 4, colors='black');
#         ax.clabel(contours, inline=True, fontsize=11)
    if plotRGrowthRateB:  # Colour by dR/dt
        nLevels_overlay = 50 if nLevels_overlay is None else nLevels_overlay
        vMat = V
        norm = MidpointNormalize(midpoint=0)
        ax.contourf(X, Y, vMat, nLevels_overlay, cmap=ListedColormap(sns.color_palette("BrBG_r", nLevels_overlay)[::-1]), alpha=0.8, norm=norm)
        #         ax.colorbar();
        contours = ax.contour(X, Y, vMat, 4, colors='black');
        ax.clabel(contours, inline=True, fontsize=11)

    ax.streamplot(X, Y, U, V, color='k', linewidth=arrowLineWidth, density=density, arrowsize=arrowsize)
    ax.set_facecolor('#E6E6E6')

    # Plot the null-clines
    s, r, cRS, cSR, p, d, D = sm.symbols('s, r, cRS, cSR, p, d, D', negative=False)
    nc_SExpr = sm.solve(sm.Eq(fX,0),r)[0]
    nc_RExpr = sm.solve(sm.Eq(fY,0),r)[1]
    nc_S = lambda sValVec: [nc_SExpr.subs([(s,sVal)]+[(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]]) for sVal in sValVec]
    nc_R = lambda sValVec: [nc_RExpr.subs([(s,sVal)]+[(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]]) for sVal in sValVec]
    # s-nullclines
    ax.plot(sVec,nc_S(sVec),lw=lw,color=sns.xkcd_rgb['forest green'],label=r'$\dot{s}=0$') #sns.xkcd_rgb['bluegreen']
    ax.plot(np.zeros_like(rVec),rVec,lw=lw,color="#107920") #sns.xkcd_rgb['bluegreen']
    # r-nullclines
    # Choose whether to plot r-nullcline as solid or dashed line
    linestyle_rNC = '--'
#     if np.any(np.isnan([float(x) for x in nc_S(sVec)])) or np.any(np.isnan([float(x) for x in nc_R(sVec)])): # Null-clines ill-defined somewhere
# #         linestyle_rNC = '--'
#         pass
#     elif np.all(np.abs(np.array(nc_R(sVec))-np.array(nc_S(sVec)))<1e-5): # Null-clines overlap
#         linestyle_rNC = '--'
    ax.plot(sVec,nc_R(sVec),lw=lw,color=sns.xkcd_rgb['bright pink'],label=r'$\dot{r}=0$',
            linestyle=linestyle_rNC) #
    ax.plot(sVec,np.zeros_like(sVec),linestyle=linestyle_rNC,lw=lw,color=sns.xkcd_rgb['bright pink']) #"#BA120A"

    # Plot the equilibrium points
    if plotSteadyStates:
        if solveNumerically:
            # Divide region under consideration into a 25x25 grid and solve for the
            # steady states numerically in this region
            startSSSearchFromHereList = [(x,y) for x in np.linspace(xlimVec[0],xlimVec[1],25)
                                     for y in np.linspace(ylimVec[0],ylimVec[1],25)]
            ssDic = {} # Dictionary to record unique steady states
            for point in startSSSearchFromHereList:
                try:
                    ss = sm.nsolve((fX.subs([(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]]),
                               fY.subs([(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]])),
                              (s,r),point)
                    ss = [np.round(float(x),2) for x in ss]
                    if ssDic.get(ss[0],True) is True:
                        ssDic[ss[0]] = ss[1]
                except:
                    pass
            equilibriaList = [(key,ssDic[key]) for key in ssDic.keys()]
        else:
            equilibriaList = sm.solve( (sm.Eq(fX, 0), sm.Eq(fY, 0)), s, r )
            equilibriaList = [equilibriaList[k] for k in range(len(equilibriaList))]#[0,1,3,2]]
        J = sm.Matrix([fX,fY])
        J = J.jacobian([s,r])
        for ss in equilibriaList:
            # Find ss
            if solveNumerically:
                sstar,rstar = ss
            else:
                sstar = ss[0].subs([(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]])
                rstar = ss[1].subs([(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]])
            # Determine stability
            localJ = J.subs([(s,ss[0]), (r,ss[1])]+[(var,paramDic[str(var)]) for var in varList if str(var) not in ["s","r"]])
            stableB = np.all([sm.re(k)<0 for k in localJ.eigenvals().keys()])
            # Plot
            if solveNumerically is False and (sstar.has('oo', '-oo', 'zoo', 'nan') or sstar.has('oo', '-oo', 'zoo', 'nan')):
                print("Warning: A steady state is complex and was not plotted")
                continue
            ax.plot(sstar,rstar,linestyle='none',marker="o",
                    markersize=30, markeredgecolor='black',
                    color=sns.xkcd_rgb['khaki green' if stableB else 'burnt red'])

    # Decorate
    ax.set_xlim(xlimVec)
    ax.set_ylim(ylimVec)
    ax.set_xlabel(r"$s$" if decorate else "")
    ax.set_ylabel(r"$s$" if decorate else "")
    ax.set_title(titleStr)
    if legendLoc!='off': ax.legend(loc=legendLoc)

    return ax

def PlotCompetitionExperiment(paramDic,initialStateVec=None,costsToPlotList=[0,0.1,0.5],
                              lineStyleList=['--','-.',':'],
                              t_end=1000,nTimePts=100,figsize=(10,8),outName=None):
    fig, currAx = plt.subplots(1,1,sharex=True,sharey=True,figsize=figsize)
    if initialStateVec is None: initialStateVec = [0.05,0.05]
    initialStateVec.append(paramDic['theta']*(initialStateVec[0] + initialStateVec[1]))
    initialStateVec.append(0)
    originalValue = paramDic['rR']
    for i,cost in enumerate(costsToPlotList):
        paramDic['rR'] = (1-cost)*originalValue
        resultsDf = Simulate_AT_FixedThreshold(modelFun=rdModel_nsKill,
                                           initialStateVec=initialStateVec,
                                           atThreshold=1.,intervalLength=t_end,
                                           paramDic=paramDic,
                                           t_end=t_end, t_eval=np.linspace(0,t_end,nTimePts))
        currAx.plot(resultsDf.Time,resultsDf.S,ls=lineStyleList[i],
                    color='#0F4C13',lw=7)
        currAx.plot(resultsDf.Time,resultsDf.R,ls='-' if cost==0 else lineStyleList[i],
                    color='#710303',lw=7)

    currAx.set_ylim([0,1.05])
    currAx.set_xlabel("")
    currAx.set_ylabel("")
    currAx.tick_params(labelsize=28)
    sns.despine(offset=5, trim=True)
    if outName is not None: plt.savefig(outName)


def PlotTTPHeatmap(dataDf,feature='RelTimeGained',cmap="Greys",vmin=0,vmax=45,cbar=True,annot=True,fmt="1.0f",ax=None):
    if ax is None: _,ax = plt.subplots(1,1,figsize=(5,5))
    initialSizeList = dataDf.InitialTumourSize.unique()
    rFracList = dataDf.RFrac.unique()
    timeGainedMat = dataDf.pivot("RFrac","InitialTumourSize",feature)
    sns.heatmap(timeGainedMat,
                vmin=vmin, vmax=vmax,
                cmap=cmap,linewidths=2,
                annot=annot,fmt=fmt,
                square=len(rFracList)==len(initialSizeList),
                cbar=cbar,ax=ax)
    ax.tick_params(labelsize=24,rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("")
# --------------------------- Miscellaneous -----------------------------------------------------
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def GenerateParameterDic(initialSize,rFrac,cost,turnover,paramDic):
    """
    Generate parameter dictionary for given initial conditions, cost and turnover values.
    :param cost:
    :param turnover:
    :param paramDic:
    :return:
    """
    # Generate ICS
    initialStateVec = [initialSize * (1 - rFrac), initialSize * rFrac, 0, paramDic['DMax']]
    initialStateVec[2] = paramDic['theta'] * (initialStateVec[0] + initialStateVec[1])

    # Generate params
    paramDic = paramDic.copy()
    paramDic['rR'] = (1-cost)*paramDic['rS']
    paramDic['dR'] = turnover*paramDic['rS']
    paramDic['dS'] = turnover*paramDic['rS']
    return initialStateVec,paramDic

