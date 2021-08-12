# ====================================================================================
# Class to simulate spheroid growth using one of 6 ODE models
# ====================================================================================
import numpy as np
import scipy.integrate
import pandas as pd
import math
import os
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import contextlib
import sys
import myUtils as utils
# ====================================================================================
class LotkaVolterraModel():
    def __init__(self, **kwargs):
        # Initialise parameters
        self.paramDic = {"rS":.027, "rR":.027, "dS":0., "dR":0., "dD":1.5, "k":1., "D":0, "theta":1, 'DMax':1.,
                         "S0":0,"R0":0}
        self.nParams = len(self.paramDic)
        self.resultsDf = None

        # Set the parameters
        self.SetParams(**kwargs)

        # Configure the solver
        self.dt = kwargs.get('dt', 1e-3)  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', 1.0e-8)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', 1.0e-6)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', 'DOP853')  # ODE solver used
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          False)  # If true, suppress output of ODE solver (including warning messages)
        self.successB = False  # Indicate successful solution of the ODE system

    # =========================================================================================
    # Function to set the parameters
    def SetParams(self, **kwargs):
        for key in self.paramDic.keys():
            self.paramDic[key] = float(kwargs.get(key, self.paramDic[key]))
        self.initialStateList = [self.paramDic['S0'], self.paramDic['R0']]

    # =========================================================================================
    # The governing equations
    def ModelEqns(self, t, uVec):
        s, r, c = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['rS']*(1 - s - r) * (1-self.paramDic['dD']*c) * s - self.paramDic['dS']*s
        dudtVec[1] = self.paramDic['rR']*(1 - (r + s)/self.paramDic['k'])*r - self.paramDic['dR']*r
        dudtVec[2] = 0
        return (dudtVec)

    # =========================================================================================
    # Function to simulate the model
    def Simulate(self, treatmentScheduleList, scaleTumourVolume=True, **kwargs):
        # Allow configuring the solver at this point as well
        self.dt = float(kwargs.get('dt', self.dt))  # Time resolution to return the model prediction on
        self.absErr = kwargs.get('absErr', self.absErr)  # Absolute error allowed for ODE solver
        self.relErr = kwargs.get('relErr', self.relErr)  # Relative error allowed for ODE solver
        self.solverMethod = kwargs.get('method', self.solverMethod) # ODE solver used
        self.successB = False  # Indicate successful solution of the ODE system
        self.suppressOutputB = kwargs.get('suppressOutputB',
                                          self.suppressOutputB)  # If true, suppress output of ODE solver (including warning messages)

        # Solve
        self.treatmentScheduleList = treatmentScheduleList
        if self.resultsDf is None or treatmentScheduleList[0][0]==0:
            currStateVec = self.initialStateList + [0]
            self.resultsDf = None
        else:
            currStateVec = [self.resultsDf['S'].iloc[-1], self.resultsDf['R'].iloc[-1], self.resultsDf['DrugConcentration'].iloc[-1]]
        resultsDFList = []
        encounteredProblemB = False
        for intervalId, interval in enumerate(treatmentScheduleList):
            tVec = np.arange(interval[0], interval[1], self.dt)
            if intervalId == (len(treatmentScheduleList) - 1):
                tVec = np.arange(interval[0], interval[1] + self.dt, self.dt)
            currStateVec[2] = interval[2]
            if self.suppressOutputB:
                with stdout_redirected():
                    solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                       t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                       method=self.solverMethod,
                                                       atol=self.absErr, rtol=self.relErr,
                                                       max_step=kwargs.get('max_step', np.inf))
            else:
                solObj = scipy.integrate.solve_ivp(self.ModelEqns, y0=currStateVec,
                                                   t_span=(tVec[0], tVec[-1] + self.dt), t_eval=tVec,
                                                   method=self.solverMethod,
                                                   atol=self.absErr, rtol=self.relErr,
                                                   max_step=kwargs.get('max_step', np.inf))
            # Check that the solver converged
            if not solObj.success or np.any(solObj.y<0):
                self.errMessage = solObj.message
                encounteredProblemB = True
                if not self.suppressOutputB: print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if not solObj.success:
                    if not self.suppressOutputB:print(self.errMessage)
                else:
                    if not self.suppressOutputB:print("Negative values encountered in the solution. Make the time step smaller or consider using a stiff solver.")
                    if not self.suppressOutputB:print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                self.solObj = solObj
                break
            # Save results
            resultsDFList.append(
                pd.DataFrame({"Time": tVec, "S": solObj.y[0, :], "R": solObj.y[1, :], "DrugConcentration": solObj.y[2, :]}))
            currStateVec = solObj.y[:, -1]
        # If the solver diverges in the first interval, it can't return any solution. Catch this here, and in this case
        # replace the solution with all zeros.
        if len(resultsDFList)>0:
            resultsDf = pd.concat(resultsDFList)
        else:
            resultsDf = pd.DataFrame({"Time": tVec, "S": np.zeros_like(tVec),
                                        "R": np.zeros_like(tVec), "DrugConcentration": np.zeros_like(tVec)})
        # Compute the tumour size that we'll see
        resultsDf['TumourSize'] = pd.Series(self.CellDensityToAreaModel(resultsDf),
                                            index=resultsDf.index)

        if self.resultsDf is not None:
            resultsDf = pd.concat([self.resultsDf,resultsDf])
        self.resultsDf = resultsDf
        if scaleTumourVolume:
            self.resultsDf['S'] /= self.resultsDf.TumourSize.iloc[0]
            self.resultsDf['R'] /= self.resultsDf.TumourSize.iloc[0]
            self.resultsDf['TumourSize'] /= self.resultsDf.TumourSize.iloc[0]
        self.successB = True if not encounteredProblemB else False

    # =========================================================================================
    # Simulate adaptive therapy
    def Simulate_AT(self, atThreshold=0.2, intervalLength=1., t_end=1000, nCycles=np.inf, t_span=None, scaleTumourVolume=True):
        t_span = t_span if t_span is not None else (0, t_end)
        currInterval = [t_span[0], t_span[0] + intervalLength]
        refSize = self.paramDic['theta']*np.sum(self.initialStateList) if self.resultsDf is None else self.resultsDf.TumourSize.iloc[-1]
        dose = self.paramDic['DMax']
        currCycleId = 0
        while (currInterval[1] <= t_end) and (currCycleId<nCycles):
            # Simulate
#             print(currInterval,refSize)
            self.Simulate([[currInterval[0], currInterval[1], dose]], scaleTumourVolume=False)

            # Update dose
            if self.resultsDf.TumourSize.iat[-1] > refSize:
                currCycleId += (dose==0)
                dose = self.paramDic['DMax']
            elif self.resultsDf.TumourSize.iat[-1] < (1-atThreshold)*refSize:
                dose = 0
            else:
                dose = (dose > 0)*self.paramDic['DMax']

            # Update interval
            currInterval = [x + intervalLength for x in currInterval]

        # Clean up the data frame
        self.resultsDf.drop_duplicates(inplace=True)
        if scaleTumourVolume:
            self.resultsDf['S'] /= self.resultsDf.TumourSize.iloc[0]
            self.resultsDf['R'] /= self.resultsDf.TumourSize.iloc[0]
            self.resultsDf['TumourSize'] /= self.resultsDf.TumourSize.iloc[0]
        return self.resultsDf
    # =========================================================================================
    # Define the model mapping cell densities to observed tumour size
    def CellDensityToAreaModel(self, popModelSolDf):
        return self.paramDic['theta']*(popModelSolDf.R.values + popModelSolDf.S.values)

    # =========================================================================================
    # Function to plot the model predictions
    def Plot(self, decoratey2=True, ax=None, **kwargs):
        if ax is None: fig, ax = plt.subplots(1,1)
        lnslist = []
        # Plot the area the we will see on the images
        if kwargs.get('plotAreaB', True):
            lnslist += ax.plot(self.resultsDf['Time'],
                                self.resultsDf['TumourSize'],
                                lw=kwargs.get('linewidthA', 4), color=kwargs.get('colorA', 'b'),
                                linestyle=kwargs.get('linestyleA', '-'), marker=kwargs.get('markerA', None),
                                label=kwargs.get('labelA', 'Model Prediction'))

        # Plot the individual populations
        if kwargs.get('plotPops', False):
            propS = self.resultsDf['S'].values / (self.resultsDf['S'].values + self.resultsDf['R'].values)
            lnslist += ax.plot(self.resultsDf['Time'],
                                propS * self.resultsDf['TumourSize'],
                                lw=kwargs.get('linewidth', 4), linestyle=kwargs.get('linestyleS', '--'),
                                color=kwargs.get('colorS', 'g'),
                                label='S')
            lnslist += ax.plot(self.resultsDf['Time'],
                                (1 - propS) * self.resultsDf['TumourSize'],
                                lw=kwargs.get('linewidth', 4), linestyle=kwargs.get('linestyleR', '--'),
                                color=kwargs.get('colorR', 'r'),
                                label='R')

            # Plot the drug concentration
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        drugConcentrationVec = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(self.resultsDf),
                                                 tVec=self.resultsDf['Time'])
        ax2.fill_between(self.resultsDf['Time'],
                         0, drugConcentrationVec, color="#8f59e0", alpha=0.2, label="Drug Concentration")
        # Format the plot
        ax.set_xlim([0, kwargs.get('xlim', 1.1*self.resultsDf['Time'].max())])
        ax.set_ylim([kwargs.get('ymin',-1.1*np.abs(self.resultsDf['TumourSize'].min())), kwargs.get('ylim', 1.1*self.resultsDf['TumourSize'].max())])
        ax2.set_ylim([0, kwargs.get('y2lim', self.resultsDf['DrugConcentration'].max()+.1)])
        ax.set_xlabel("Time")
        ax.set_ylabel("Tumour Size")
        ax2.set_ylabel(r"Drug Concentration in $\mu M$" if decoratey2 else "")
        ax.set_title(kwargs.get('title', ''))
        if kwargs.get('plotLegendB', True):
            labsList = [l.get_label() for l in lnslist]
            plt.legend(lnslist, labsList, loc=kwargs.get('legendLoc', "upper right"))
        plt.tight_layout()
        if kwargs.get('saveFigB', False):
            plt.savefig(kwargs.get('outName', 'modelPrediction.png'), orientation='portrait', format='png')
            plt.close()
        if kwargs.get('returnAx', False): return ax


# ====================================================================================
# Functions used to suppress output from odeint
# Taken from: https://stackoverflow.com/questions/31681946/disable-warnings-originating-from-scipy
def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
