// ========================================================================
// Class to hold the experiments I ran to explore AT in 2-d.
// ========================================================================

package onLatticeCA;
import Framework.Util;
import com.sun.org.apache.xpath.internal.operations.Bool;

import java.io.File;
import java.util.Arrays;

public class runParameterSweep {
    public static void main(String[] args) {
        /* ========================================================================
         * --- Parameters ----
         * ======================================================================== */
        int xDim = 100; // x-Dimensions of the grid
        int yDim = 100; // y-Dimensions of the grid
        int[] initPopSizeArr = {0,0}; // Initial population sizes
        double divisionRate_S = .027;
        double divisionRate_R = divisionRate_S;
        double deathRate_S = 0;
        double deathRate_R = 0;
        double movementRate = 0.;
        double drugKillProportion = 0.75; // Corresponds to 1.5 from ODE
        String initialSeedingType = "random"; // How cells are initially distributed. Options: "random", "separation1d", "separation2d", "circle"
        int initialSeedingDistance = 10; // Distance which cells are seed apart if seeded using "separation_x", or radius of circle if seeded as "circle"
        Boolean compareToMTD = true; // Compare AT to MTD. If false simulate only the single treatment given by atThreshold.
        double atThreshold = 0.5; // Relative size reduction when treatment is withdrawn under AT
        Boolean simulateSpecificSchedule = false; // Whether to simulate a specific pre-defined Tx schedule
        double[][] treatmentScheduleList = null; // Treatment schedule to simulate
        Boolean predictIntermittentTherapyOutcome = false; // Whether to predict the outcome according to the Bruchovsky et al (2006) intermittent schedule
        double initialPSA = 1; // Baseline PSA value. Used for determining normal and elevated ranges during intermittent schedule
        int weeksOnTreatment = 0; // Weeks on treatment prior to start of intermittent therapy prediction
        int[] previousFourMeasurements = new int[] {0,0,0,0}; // If patient had been receiving therapy, these are their last four measurements
        Boolean simulateCT = false; // Whether to simulate IMT or CT
        double proProlifFac = 0; // Factor by which growth rate is increased during off-treatment phases
        double proProlifBalance = 1; // Relative effect of pro proliferation treatment on resistant cells
        double proTurnoverFac = 0; // Relative increase in death rate in 'off-treatment' phase due to low-dose chemo
        double proTurnoverBalance = 1; // Relative effect of pro turnover treatment on resistant cells
        double[][] proTurnoverSchedule=null; // Schedule of pro-turnover drug. Used to obtain controls for pro-turnover experiments.
        double dt = 1.;
        double tEnd = 10000; // End time
        double[] paramArr;
        int nReplicates = 1000;
        String outDir = "./tmp/";

        // Helper variables
        OnLatticeCA myModel = new OnLatticeCA();
        String outFName;
        String txName;
        double initialSizeProp=-1;
        int initialSize=-1;
        double rFrac=-1;
        double cost=-1;
        Boolean profilingMode=true;
        Boolean terminateAtProgression = true;
        int seed=-1;
        String imageOutDir="";
        int imageFreq=10;

        // Parse command line inputs
        for (int i = 0; i < args.length; i+=2) {
            if (args[i].equalsIgnoreCase("-xDim")) {
                xDim = Integer.parseInt(args[i + 1]);
            }
            if (args[i].equalsIgnoreCase("-yDim")) {
                yDim = Integer.parseInt(args[i + 1]);
            }
            if (args[i].equalsIgnoreCase("-initialSize")) {
                initialSizeProp = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-rFrac")) {
                rFrac = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-turnover")) {
                deathRate_S = Double.parseDouble(args[i + 1]) * divisionRate_S;
                deathRate_R = Double.parseDouble(args[i + 1]) * divisionRate_S;
            }
            else if (args[i].equalsIgnoreCase("-cost")) {
                cost = Double.parseDouble(args[i + 1]);
                divisionRate_R = (1 - cost) * divisionRate_S;
            }
            else if (args[i].equalsIgnoreCase("-initialSeedingType")) {
                initialSeedingType = args[i + 1];
            }
            else if (args[i].equalsIgnoreCase("-initialSeedingDistance")) {
                initialSeedingDistance = Integer.parseInt(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-compareToMTD")) {
                compareToMTD= Boolean.parseBoolean(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-atThreshold")) {
                atThreshold = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-simulateSpecificSchedule")) {
                simulateSpecificSchedule = Boolean.parseBoolean(args[i + 1]);
                compareToMTD = false;
            }
            else if (args[i].equalsIgnoreCase("-treatmentScheduleList")) {
                String[] tmpList = args[i+1].split("\\[");
                String[] processed_justNumbers;
                treatmentScheduleList = new double[tmpList.length][3];
                for (int k=2; k<tmpList.length; k++) {
                    processed_justNumbers = tmpList[k].split("\\]");
                    processed_justNumbers = processed_justNumbers[0].split(",");
                    for (int kk=0; kk<processed_justNumbers.length; kk++) {
                        treatmentScheduleList[k-2][kk] = Double.parseDouble(processed_justNumbers[kk]);
                    }
                }
            }
            else if (args[i].equalsIgnoreCase("-predictIntermittentTherapyOutcome")) {
                predictIntermittentTherapyOutcome = Boolean.parseBoolean(args[i + 1]);
                simulateSpecificSchedule = predictIntermittentTherapyOutcome? true:simulateSpecificSchedule;
                compareToMTD = false;
            }
            else if (args[i].equalsIgnoreCase("-initialPSA")) {
                initialPSA = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-weeksOnTreatment")) {
                weeksOnTreatment = Integer.parseInt(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-previousFourMeasurements")) {
                String[] tmpList = args[i+1].split("\\[");
                String[] processed_justNumbers;
                previousFourMeasurements = new int[4];
                for (int k=2; k<tmpList.length; k++) {
                    processed_justNumbers = tmpList[k].split("\\]");
                    processed_justNumbers = processed_justNumbers[0].split(",");
                    for (int kk=0; kk<processed_justNumbers.length; kk++) {
                        previousFourMeasurements[kk] = Integer.parseInt(processed_justNumbers[kk]);
                    }
                }
            }
            else if (args[i].equalsIgnoreCase("-simulateCT")) {
                simulateCT = Boolean.parseBoolean(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-proProlifFac")) {
                proProlifFac = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-proProlifBalance")) {
                proProlifBalance = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-proTurnoverFac")) {
                proTurnoverFac = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-proTurnoverBalance")) {
                proTurnoverBalance= Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-proTurnoverSchedule")) {
                String[] tmpList = args[i+1].split("\\[");
                String[] processed_justNumbers;
                proTurnoverSchedule = new double[tmpList.length][3];
                for (int k=2; k<tmpList.length; k++) {
                    processed_justNumbers = tmpList[k].split("\\]");
                    processed_justNumbers = processed_justNumbers[0].split(",");
                    for (int kk=0; kk<processed_justNumbers.length; kk++) {
                        proTurnoverSchedule[k-2][kk] = Double.parseDouble(processed_justNumbers[kk]);
                    }
                }
            }
            else if (args[i].equalsIgnoreCase("-tEnd")) {
                tEnd = Double.parseDouble(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-seed")) {
                seed = Integer.parseInt(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-nReplicates")) {
                nReplicates = Integer.parseInt(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-outDir")) {
                outDir = args[i + 1];
            }
            else if (args[i].equalsIgnoreCase("-profilingMode")) {
                profilingMode = Boolean.parseBoolean(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-terminateAtProgression")) {
                terminateAtProgression = Boolean.parseBoolean(args[i + 1]);
            }
            else if (args[i].equalsIgnoreCase("-imageOutDir")) {
                imageOutDir = args[i + 1];
            }
            else if (args[i].equalsIgnoreCase("-imageFreq")) {
                imageFreq = Integer.parseInt(args[i + 1]);
            }
        }
//        System.out.println(Util.ArrToString(new double[] {initialSizeProp,rFrac,deathRate_S,cost,tEnd,nReplicates},","));

        // Run the sweep
        System.setProperty("java.awt.headless", "true");
        double[] treatmentList = new double[] {atThreshold};
        if (compareToMTD){treatmentList=new double[] {1., atThreshold};}
        int[] replicateIdList = new int[] {seed};
        int replicateId;
        if (nReplicates>1) {
            replicateIdList = new int[nReplicates];
            for (int i=0; i<nReplicates; i++) {replicateIdList[i]=i;}
        }
        initialSize = (int) (initialSizeProp * xDim * yDim);
        initPopSizeArr = new int[]{(int) Math.floor(initialSize * (1 - rFrac)), (int) Math.ceil(initialSize * rFrac)};
        new File(outDir).mkdirs();
        for (int replicateIdx = 0; replicateIdx < nReplicates; replicateIdx++) {
            replicateId = replicateIdList[replicateIdx];
            for (int txId = 0; txId < treatmentList.length; txId++) {
                // Set up the simulation
                if (simulateSpecificSchedule) {
                    txName = "results";
                } else {
                    txName = treatmentList[txId] == 1. ? "MTD" : "AT"+(int) (treatmentList[txId]*100);
                }
                paramArr = new double[]{divisionRate_S, divisionRate_R, movementRate, movementRate,
                        deathRate_S, deathRate_R, drugKillProportion};
                myModel = new OnLatticeCA(xDim, yDim, paramArr, dt);
                myModel.tEnd = tEnd;
                myModel.terminateAtProgression = terminateAtProgression;
                myModel.visualiseB = false;
                myModel.verboseLevel = 0;

                // Set the random number seed
                if (seed!=-1) {
                    if (nReplicates==1) {myModel.SetSeed(seed);}
                    else {myModel.SetSeed(replicateId,seed);}
                } else {myModel.SetSeed(replicateId);}

                // Set the logging behaviour
                if (proProlifFac!=0) {
                    myModel.SetExtraSimulationInfo(new String[]{"ReplicateId", "InitSize", "RFrac", "TxName", "Cost", "ProProlifFac", "ProProlifBalance"},
                            new double[]{replicateId, initialSizeProp, rFrac, txId, cost, proProlifFac, proProlifBalance});
                } else if(proTurnoverFac!=0) {
                    myModel.SetExtraSimulationInfo(new String[]{"ReplicateId", "InitSize", "RFrac", "TxName", "Cost", "ProTurnoverFac", "ProTurnoverBalance"},
                            new double[]{replicateId, initialSizeProp, rFrac, txId, cost, proTurnoverFac, proTurnoverBalance});
                }
                else {
                    myModel.SetExtraSimulationInfo(new String[]{"ReplicateId", "InitSize", "RFrac", "TxName", "Cost"},
                            new double[]{replicateId, initialSizeProp, rFrac, txId, cost});
                }
                outFName = outDir + txName + "_cellCounts_cost_" + cost*100 + "_rFrac_" + rFrac + "_initSize_" + initialSizeProp + "_dt_" + dt + "_RepId_" + replicateId + ".csv";
                if (simulateSpecificSchedule) {outFName = outDir + txName + "_RepId_" + replicateId + ".csv";}
                if (imageOutDir.length()>0) {
                    myModel.visualiseB = true;
                    myModel.pause = 0;
                    String currImgOutDir = imageOutDir+txName + "_cost_" + cost*100 + "_rFrac_" + rFrac + "_initSize_" + initialSizeProp + "_dt_" + dt + "_RepId_" + replicateId;
                    if (simulateSpecificSchedule) {currImgOutDir = imageOutDir + txName + "_RepId_" + replicateId;}
                    new File(currImgOutDir).mkdirs();
                    myModel.ConfigureImaging(currImgOutDir, imageFreq);
                }

                // Initialise the simulation
                myModel.InitialiseCellLog(outFName, dt, profilingMode);
                if (initialSeedingType.equalsIgnoreCase("random")) {
                    myModel.SetInitialState(initPopSizeArr);
                } else {
                    if (initialSeedingType.equalsIgnoreCase("separation1d")) {
                        initPopSizeArr = new int[]{initialSize - 8, 8};
                    }
                    myModel.SetInitialState(initPopSizeArr, initialSeedingType, initialSeedingDistance);
                }

                // Run the simulation
                if (proProlifFac!=0) {myModel.ProfileProProlifAT(treatmentList[txId], dt, proProlifFac, proProlifBalance);}
                else if(proTurnoverSchedule!=null) {myModel.RunProTurnoverControl(proTurnoverSchedule, proTurnoverFac, proTurnoverBalance);}
                else if(proTurnoverFac!=0) {myModel.ProfileProTurnoverAT(treatmentList[txId], dt, proTurnoverFac, proTurnoverBalance);}
                else if(predictIntermittentTherapyOutcome) {myModel.SetTreatmentSchedule(treatmentScheduleList); myModel.PredictIntermittentSchedule(initialPSA,weeksOnTreatment,previousFourMeasurements,simulateCT);}
                else if(simulateSpecificSchedule) {myModel.SetTreatmentSchedule(treatmentScheduleList); myModel.Run();}
                else {myModel.ProfileAT(treatmentList[txId], dt);}
                myModel.Close();
            }
        }
    }
}
