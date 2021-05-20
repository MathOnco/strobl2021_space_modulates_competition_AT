package onLatticeCA;

import Framework.GridsAndAgents.AgentGrid2D;
import Framework.Gui.GridWindow;
import Framework.Gui.UIGrid;
import Framework.Rand;
import Framework.Tools.FileIO;
import Framework.Util;
import com.sun.org.apache.xpath.internal.operations.Bool;

import java.util.Arrays;

public class OnLatticeCA extends AgentGrid2D<Cell> {
    // Simulation parameters
    int[] initPopSize; // Initial population sizes
    double divisionRate_S; // Proliferation rate of sensitive cells in d^-1. Proliferation will be attempted at this rate.
    double divisionRate_R; // Proliferation rate of resistant cells in d^-1
    double movementRate_S; // Movement rate of sensitive cells in d^-1. Movement will be attempted at this rate.
    double movementRate_R; // Movement rate of resistant cells in d^-1.
    double deathRate_S; // Natural death rate of sensitive cells in d^-1. Mean life span of a cell will be 1/k
    double deathRate_R;// Natural death rate of resistant cells in d^-1.
    double drugKillProportion; // Drug induced death rate in d^-1.
    double currDrugConcentration; // Current drug concentration in [0,1].
    double[][] treatmentScheduleList; // Treatment schedule in format {{tStart, tEnd, drugConcentration}}
    double tEnd; // End time in d
    double dt = 0.01; // Time step in d
    int nCycles = 0; // Number of completed AT cycles
    int nTSteps; // Number of time steps to simulate
    int[] hood = Util.VonNeumannHood(false); // Define division in von Neumann neighbourhood. Argument indicates that we don't want the center position returned (can only divide into neighbouring squares).
    String initialConditionType = "random";
    double initRadius = 10.; // Radius circle if cells seeded as circle
    int dist = 0; // Distance of the two nests in the distance initial conditions
    int nFailedDivs_R = 0;
    int nAttemptedDivs_R = 0;
    int nDeath_R = 0;

    // Helper variables
    int tIdx = 0;
    int scaleFactor = 2; // Scale factor used to display grid
    int pause = 0; // Pause betwween time steps to smoothen simulation
    int[] cellCountsArr = new int[2]; // Array to hold number of each cell type
    double[] paramArr;
    Rand rn = new Rand(1); // Random number generator
    Rand rn_ICs = new Rand(1); // Random number generator for initial conditions
    UIGrid vis;
    final static int BLACK = Util.RGB(0, 0, 0);

    // Output
    int verboseLevel = 2;
    Boolean terminateAtProgression = true; // Terminates AT profiling at progression. If not it will be terminated at tEnd
    double printFrequency = 1; // Frequency at which output is printed to the screen
    String cellCountLogFileName;
    FileIO cellCountLogFile = null;
    double logCellCountFrequency = -1; // Frequency (in time units) at which the cell counts are written to file. <0 indicates no logging
    Boolean profilingMode = false; // Whether to log directly to file or save in memory until the end of the simulation
    double[][] outputArr; // Array to hold the simulation results if they're not directly logged to file
    Boolean visualiseB = true;
    int imageFrequency = -1; // Frequency at which an image of the tumour is saved. Negative number turns it off
    private String imageOutDir = "."; // Directory which to save images to
    double[] extraSimulationInfo;
    String[] extraSimulationInfoNames;

    // ------------------------------------------------------------------------------------------------------------
    public static void main(String[] args) {
        double[] paramArr = new double[]{.1, .1, 0., 0.,
                0., 0., 1.};
        OnLatticeCA myModel = new OnLatticeCA(100, 100, paramArr, 0.01);
//        myModel.tEnd = 10;
        myModel.dt = 1;
        myModel.visualiseB = true;
        myModel.verboseLevel = 2;
        myModel.initPopSize = new int[] {5000,2};
        myModel.SetSeed(42);
        myModel.SetDrugConcentration(0);
        myModel.SetTreatmentSchedule(new double[][]{{0,30,1},{30,60,0}});
        myModel.printFrequency = 1e2;
        myModel.ConfigureImaging("./",30);

        // Simulate
//        myModel.SetInitialState(new int[] {50,50}, "random",25);
        myModel.InitialiseCellLog("tst.csv",1,true);
        myModel.PredictIntermittentSchedule(15, 0, new int[] {0, 0, 0, 0}, false);
        myModel.Close();

//        GridWindow currVis=new GridWindow(myModel.xDim,myModel.yDim,myModel.scaleFactor,false,null, myModel.visualiseB);//used for visualization
//        myModel.vis = currVis;
//        myModel.InitSimulation_Separation2d(48,10);
//        myModel.Close();

    }

    // ------------------------------------------------------------------------------------------------------------
    // Grid Constructors
    public OnLatticeCA() {
        super(100, 100, Cell.class);
    }

    public OnLatticeCA(int x, int y, double[] paramArr, double dt) {
        super(x, y, Cell.class);
        SetParameters(paramArr);
        this.dt = dt;
    }

    public OnLatticeCA(int x, int y, double[] paramArr, double dt, String cellCountLogFileName) {
        super(x, y, Cell.class);
        SetParameters(paramArr);
        this.dt = dt;
        this.cellCountLogFileName = cellCountLogFileName;
    }

    public OnLatticeCA(int x, int y, double[] paramArr, double dt, UIGrid vis) {
        super(x, y, Cell.class);
        SetParameters(paramArr);
        this.dt = dt;
        this.vis = vis;
    }

    public OnLatticeCA(int x, int y, double[] paramArr, double dt, UIGrid vis, String cellCountLogFileName) {
        super(x, y, Cell.class);
        SetParameters(paramArr);
        this.dt = dt;
        this.vis = vis;
        this.cellCountLogFileName = cellCountLogFileName;
    }

    // ------------------------------------------------------------------------------------------------------------
    // Functions to parameterise the model
    public void SetParameters(double[] paramArr) {
        this.divisionRate_S = paramArr[0];
        this.divisionRate_R = paramArr[1];
        this.movementRate_S = paramArr[2];
        this.movementRate_R = paramArr[3];
        this.deathRate_S = paramArr[4];
        this.deathRate_R = paramArr[5];
        this.drugKillProportion = paramArr[6];
    }

    public void SetInitialState(int[] initialStateArr) {this.initPopSize = initialStateArr;}

    public void SetInitialState(int[] initialStateArr, Boolean initialiseRandom, double initRadius) {
        this.initPopSize = initialStateArr;
        this.initialConditionType = initialiseRandom? "random":"circle";
        this.initRadius = initRadius;
    }

    public void SetInitialState(int[] initialStateArr, String initialConditionType, int x) {
        this.initPopSize = initialStateArr;
        this.initialConditionType = initialConditionType;
        if (initialConditionType.equals("circle")) {this.initRadius = x;}
        else if (initialConditionType.equals("separation1d") || initialConditionType.equals("separation2d")) {this.dist = x;}
    }

    public void SetInitialState(int[] initialStateArr, String initialConditionType, int dist, int seed) {
        this.initPopSize = initialStateArr;
        this.initialConditionType = initialConditionType;
        if (initialConditionType.equals("separation1d") || initialConditionType.equals("separation2d")) {
            this.dist = dist;
            this.rn_ICs = new Rand(seed);
        }
    }

    public void SetDrugConcentration(double drugConcentration) {
        this.currDrugConcentration = drugConcentration;
    }

    public void SetTreatmentSchedule(double[][] treatmentScheduleList) {
        this.treatmentScheduleList = treatmentScheduleList;
    }

    public void SetSeed(int seed) {
        this.rn = new Rand(seed);
        this.rn_ICs = new Rand(seed);
    }

    public void SetSeed(int seed_Simulation, int seed_ICs) {
        this.rn = new Rand(seed_Simulation);
        this.rn_ICs = new Rand(seed_ICs);
    }

    public void SetVerboseness(int verboseLevel, double printFrequency) {
        this.verboseLevel = verboseLevel;
        this.printFrequency = printFrequency;
    }

    public void ConfigureVisualisation(boolean visualiseB, int pause) {
        this.visualiseB = visualiseB;
        this.pause = pause;
    }

    public void ConfigureImaging(String imageOutDir, int imageFrequency) {
        /*
        * Configure location of and frequency at which tumour is imaged.
        */
        this.imageOutDir = imageOutDir;
        this.imageFrequency = imageFrequency;
    }

    public double[] GetParameters() {
        return new double[] {divisionRate_S, divisionRate_R, movementRate_S, movementRate_R,
                deathRate_S, deathRate_R, drugKillProportion};
    }

    public double[] GetModelState() {
        return new double[] {tIdx, tIdx*dt, cellCountsArr[0], cellCountsArr[1], Util.ArraySum(cellCountsArr),
                currDrugConcentration, divisionRate_S, divisionRate_R, movementRate_S, movementRate_R,
                deathRate_S, deathRate_R, drugKillProportion, dt, nCycles, nAttemptedDivs_R, nFailedDivs_R, nDeath_R};
    }

    // ------------------------------------------------------------------------------------------------------------
    // Seeding functions for the cells
    public void InitSimulation_Random(int s0, int r0) {
        //places tumor cells randomly on the dish
        int[] distributionOfResistanceArr = new int[s0 + r0];
        int[] initialLocArr = new int[xDim * yDim];
        Arrays.fill(cellCountsArr, 0); // clear the cell counter

        // Generate a list of random grid positions
        for (int i = 0; i < xDim * yDim; i++) {
            initialLocArr[i] = i;
        }
        rn_ICs.Shuffle(initialLocArr);

        // Generate a list of random assignment to sensitive or resistance
        for (int i = 0; i < s0; i++) {
            distributionOfResistanceArr[i] = 0;
        }
        for (int i = s0; i < s0 + r0; i++) {
            distributionOfResistanceArr[i] = 1;
        }
        rn_ICs.Shuffle(distributionOfResistanceArr);

        // Create cells according to the above assignments
        for (int i = 0; i < s0 + r0; i++) {
            Cell c = NewAgentSQ(initialLocArr[i]);
            c.resistance = distributionOfResistanceArr[i];
            if (c.resistance == 0) {
                c.divisionRate = this.divisionRate_S;
                c.movementRate = this.movementRate_S;
                c.deathRate = this.deathRate_S;
            } else {
                c.divisionRate = this.divisionRate_R;
                c.movementRate = this.movementRate_R;
                c.deathRate = this.deathRate_R;
            }
            c.Draw();
            cellCountsArr[c.resistance] += 1; // Update population counter
        }
    }

    public void InitSimulation_Separation1d(int dist, int nCells_S) {
        // places resistant cells in two squares a distance dist apart, and nCells_S sensitive cells
        // randomly around them (seed: seed_ICs).
        Arrays.fill(cellCountsArr, 0); // clear the cell counter

        // 1. Seed the resistant cells
        int[] currTopLeftCornerPos;
        int offset;
        for (int clusterId = 0; clusterId < 2; clusterId++) {
            offset = clusterId==0? -2-dist/2: dist/2;
            currTopLeftCornerPos = new int[] {xDim/2+offset,yDim/2};
            for (int i = 0; i < 4; i++) {
                Cell c = NewAgentSQ(currTopLeftCornerPos[0]+i%2,currTopLeftCornerPos[1]+i/2);
                c.resistance = 1;
                c.divisionRate = this.divisionRate_R;
                c.movementRate = this.movementRate_R;
                c.deathRate = this.deathRate_R;
                c.Draw();
                cellCountsArr[c.resistance] += 1; // Update population counter
            }
        }

        // 2. Seed the sensitive cells
        int[] initialLocArr = new int[xDim * yDim];
        for (int i = 0; i < xDim * yDim; i++) {
            initialLocArr[i] = i;
        }
        rn_ICs.Shuffle(initialLocArr);
        int positionArrIdx = 0;
        Boolean placedAgent = false;
        for (int i = 0; i < nCells_S; i++) {
            /** Try/catch structure in place to catch the case when accidentally a sensitive cell is placed
             * on top of a resistant cell. If this happens, we move on to the next place in the (random) array.
             */
            placedAgent = false;
            while (!placedAgent) {
                try {
                    Cell c = NewAgentSQ(initialLocArr[positionArrIdx]);
                    c.resistance = 0;
                    c.divisionRate = this.divisionRate_S;
                    c.movementRate = this.movementRate_S;
                    c.deathRate = this.deathRate_S;
                    c.Draw();
                    cellCountsArr[c.resistance] += 1; // Update population counter
                    placedAgent = true;
                } catch (RuntimeException e) {}
                positionArrIdx++;
            }
        }
    }

    public void InitSimulation_Separation2d(int dist, int nCells_S) {
        // places resistant cells in four squares a distance dist apart, and nCells_S sensitive cells
        // randomly around them (seed: seed_ICs).
        Arrays.fill(cellCountsArr, 0); // clear the cell counter

        // 1. Seed the resistant cells
        int[] currTopLeftCornerPos;
        int offset_x;
        int offset_y;
        for (int clusterId = 0; clusterId < 4; clusterId++) {
            offset_x = clusterId/2==0? -2-dist/2: dist/2;
            offset_y = clusterId%2==0? -2-dist/2: dist/2;
            currTopLeftCornerPos = new int[] {xDim/2+offset_x,yDim/2+offset_y};
            for (int i = 0; i < 4; i++) {
                Cell c = NewAgentSQ(currTopLeftCornerPos[0]+i%2,currTopLeftCornerPos[1]+i/2);
                c.resistance = 1;
                c.divisionRate = this.divisionRate_R;
                c.movementRate = this.movementRate_R;
                c.deathRate = this.deathRate_R;
                c.Draw();
                cellCountsArr[c.resistance] += 1; // Update population counter
            }
        }

        // 2. Seed the sensitive cells
        int[] initialLocArr = new int[xDim * yDim];
        for (int i = 0; i < xDim * yDim; i++) {
            initialLocArr[i] = i;
        }
        rn_ICs.Shuffle(initialLocArr);
        int positionArrIdx = 0;
        Boolean placedAgent = false;
        for (int i = 0; i < nCells_S; i++) {
            /** Try/catch structure in place to catch the case when accidentally a sensitive cell is placed
             * on top of a resistant cell. If this happens, we move on to the next place in the (random) array.
             */
            placedAgent = false;
            while (!placedAgent) {
                try {
                    Cell c = NewAgentSQ(initialLocArr[positionArrIdx]);
                    c.resistance = 0;
                    c.divisionRate = this.divisionRate_S;
                    c.movementRate = this.movementRate_S;
                    c.deathRate = this.deathRate_S;
                    c.Draw();
                    cellCountsArr[c.resistance] += 1; // Update population counter
                    placedAgent = true;
                } catch (RuntimeException e) {}
                positionArrIdx++;
            }
        }
    }

    public void InitSimulation_Circle(double radius, double rFrac) {
        //places tumor cells in a circle
        int[] circleHood_all = Util.CircleHood(true, radius);
        int[] circleHood_resistant = Util.CircleHood(true, Math.sqrt(rFrac)*radius); // Inner circle for the resistant cells
        int nCells = MapHood(circleHood_all, xDim / 2, yDim / 2);
        int nCell_R = MapHood(circleHood_resistant, xDim / 2, yDim / 2);
//        int[] distributionOfResistanceArr = new int[len];
        Arrays.fill(cellCountsArr, 0); // clear the cell counter

        // Fill the inner circle with resistant cells
        for (int i = 0; i < nCell_R; i++) {
            Cell c = NewAgentSQ(circleHood_resistant[i]);
            c.resistance = 1;
            c.divisionRate = this.divisionRate_R;
            c.movementRate = this.movementRate_R;
            c.deathRate = this.deathRate_R;
            c.Draw();
            cellCountsArr[c.resistance] += 1; // Update population counter
        }

        // Fill the outer circle with sensitive cells
        // This is a bit dirty but works. Basically try to place a sensitive cell at all locations
        // within the big circle. If a spot is already taken because there's a resistant cell there
        // then just skip this iteration.
        for (int i = 0; i < nCells; i++) {
            try {
                Cell c = NewAgentSQ(circleHood_all[i]);
                c.resistance = 0;
                c.divisionRate = this.divisionRate_S;
                c.movementRate = this.movementRate_S;
                c.deathRate = this.deathRate_S;
                c.Draw();
                cellCountsArr[c.resistance] += 1;
            } catch (RuntimeException e) {}
        }
    }
//    public void InitSimulation_Circle(double radius, double rFrac) {
//        //places tumor cells in a circle
//        int[] circleHood = Util.CircleHood(true, radius); //generate circle neighborhood [x1,y1,x2,y2,...]
//        int len = MapHood(circleHood, xDim / 2, yDim / 2);
//        int[] distributionOfResistanceArr = new int[len];
//        Arrays.fill(cellCountsArr, 0); // clear the cell counter
//
//        // Generate a list of random assignment to sensitive or resistance
//        for (int i = 0; i < (int) ((1-rFrac)*len); i++) {
//            distributionOfResistanceArr[i] = 0;
//        }
//        for (int i = (int) ((1-rFrac)*len); i < (int) len; i++) {
//            distributionOfResistanceArr[i] = 1;
//        }
//        rn.Shuffle(distributionOfResistanceArr);
//
//        for (int i = 0; i < len; i++) {
//            Cell c = NewAgentSQ(circleHood[i]);
//            c.resistance = distributionOfResistanceArr[i];
//            if (c.resistance == 0) {
//                c.divisionRate = this.divisionRate_S;
//                c.movementRate = this.movementRate_S;
//                c.deathRate = this.deathRate_S;
//            } else {
//                c.divisionRate = this.divisionRate_R;
//                c.movementRate = this.movementRate_R;
//                c.deathRate = this.deathRate_R;
//            }
//            c.Draw();
//            cellCountsArr[c.resistance] += 1; // Update population counter
//        }
//    }

    // ------------------------------------------------------------------------------------------------------------
    public void StepCells() {
        Arrays.fill(cellCountsArr, 0);//clear the cell counts
        double totPropensity;
        int currPos;
        boolean moveSuccessB;
        double deathProb;
        double r;
        nFailedDivs_R = 0;
        nAttemptedDivs_R = 0;
        nDeath_R = 0;
        for (Cell c : this) { //iterate over all cells in the grid
            // Compute the total propensity function (the probability of an event happening)
            totPropensity = (c.divisionRate + c.movementRate +
                    c.deathRate) * dt;

            // Check if an event occured
            if (rn.Double() < totPropensity) {
                r = rn.Double();
                // 1. Division
                if (r < ((c.divisionRate * dt) / totPropensity)) {
                    nAttemptedDivs_R += c.resistance;
                    if (c.HasSpace()) {
                        // If drug is present, check if cell is killed
                        deathProb = drugKillProportion*currDrugConcentration*(1-c.resistance); // Note the kill rate is technically a proportion
                        if (rn.Double() < deathProb) {
                            vis.SetPix(c.Isq(), BLACK); // Death
                            c.Dispose();// Removes cell from spatial grid and iteration
                            cellCountsArr[c.resistance] -= 1; // Update population counter
                        } else {
                            c.Divide();
                            cellCountsArr[c.resistance] += 1;
                        }
                    } else {
                        nFailedDivs_R += c.resistance;}
                // 2. Movement
                } else if (r < ((c.divisionRate + c.movementRate) * dt) / totPropensity) {
                    currPos = c.Isq();
                    moveSuccessB = c.Move();
                    if (moveSuccessB) {
                        vis.SetPix(currPos, BLACK); // Remove from old position
                        c.Draw(); // Draw in new place
                    }
                // 3. Natural Death
                } else {
                    vis.SetPix(c.Isq(), BLACK); // Death
                    c.Dispose();// Removes cell from spatial grid and iteration
                    cellCountsArr[c.resistance] -= 1; // Update population counter
                    nDeath_R += c.resistance;
                }
            }
            cellCountsArr[c.resistance] += 1; // Update population counter
        }
        ShuffleAgents(rn); //shuffle order of for loop iteration over cells
    }

    // ------------------------------------------------------------------------------------------------------------
    public void Run() {
        // Initialise visualisation window
//        GridWindow currVis=new GridWindow(xDim,yDim,scaleFactor,false,null, visualiseB);//used for visualization
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        Boolean logged = false;
        // Set up the grid and initialise log if this is the beginning of the simulation
        if (tIdx==0) {
            if (initialConditionType.equalsIgnoreCase("random")) {
                InitSimulation_Random(initPopSize[0],initPopSize[1]);
            } else if (initialConditionType.equalsIgnoreCase("separation1d")) {
                InitSimulation_Separation1d(dist, initPopSize[0]);
            } else if (initialConditionType.equalsIgnoreCase("separation2d")) {
                InitSimulation_Separation2d(dist,initPopSize[0]);
            } else if (initialConditionType.equalsIgnoreCase("circle")) {
                InitSimulation_Circle(initRadius,((double) initPopSize[1])/(initPopSize[0]+initPopSize[1]));
            }
            PrintStatus(0);
            if (cellCountLogFile==null && cellCountLogFileName!=null) {InitialiseCellLog(this.cellCountLogFileName);}
            SaveCurrentCellCount(0);
            SaveTumourImage(tIdx);
            tIdx = 1;
        } else {
            // Continue from a restart
            if (cellCountLogFileName!=null) {
                cellCountLogFile = new FileIO(cellCountLogFileName, "a");
            }
            for (Cell c : this) {
                c.Draw();
            }
        }

        // Run the simulation
        double currIntervalEnd;
        int initialCellNumber = Util.ArraySum(cellCountsArr);
        if (treatmentScheduleList==null) treatmentScheduleList = new double[][]{{0,tEnd,currDrugConcentration}};
        for (int intervalIdx=0; intervalIdx<treatmentScheduleList.length;intervalIdx++) {
            currIntervalEnd = treatmentScheduleList[intervalIdx][1];
            nTSteps = (int) Math.ceil(currIntervalEnd/dt);
            currDrugConcentration = treatmentScheduleList[intervalIdx][2];
            completedSimulationB = false;
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
//                StepCells_DivDeath();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                // Check if the stopping condition is met
                completedSimulationB = (tIdx>nTSteps)?true:false;
//                completedSimulationB = (Util.ArraySum(cellCountsArr)>1.21*initialCellNumber) && (tIdx*dt>150);
    //            completedSimulationB = Util.ArraySum(cellCountsArr)<100;
//                completedSimulationB = cellCountsArr[0]<1 | cellCountsArr[1]>0;
//                completedSimulationB = cellCountsArr[1]>7500;
            }
        }

        // Close the simulation
//        currVis.Close();
        this.Close(logged);
    }

    // ------------------------------------------------------------------------------------------------------------
    public void ProfileAT(double atThreshold, double intervalLength) {
        // Initialise visualisation window
//        GridWindow currVis = new GridWindow(xDim,yDim,scaleFactor,false,null, visualiseB);//used for visualization
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        currDrugConcentration = 1.;

        // Set up the grid and initialise log if this is the beginning of the simulation
        if (tIdx==0) {
            if (initialConditionType.equalsIgnoreCase("random")) {
                InitSimulation_Random(initPopSize[0],initPopSize[1]);
            } else if (initialConditionType.equalsIgnoreCase("separation1d")) {
                InitSimulation_Separation1d(dist, initPopSize[0]);
            } else if (initialConditionType.equalsIgnoreCase("separation2d")) {
                InitSimulation_Separation2d(dist,initPopSize[0]);
            } else if (initialConditionType.equalsIgnoreCase("circle")) {
                InitSimulation_Circle(initRadius,((double) initPopSize[1])/(initPopSize[0]+initPopSize[1]));
            }
            PrintStatus(0);
            if (cellCountLogFile==null && cellCountLogFileName!=null) {InitialiseCellLog(this.cellCountLogFileName);}
            SaveCurrentCellCount(0);
            SaveTumourImage(tIdx);
            tIdx = 1;
        } else {
            // Continue from a restart
            if (cellCountLogFileName!=null) {
                cellCountLogFile = new FileIO(cellCountLogFileName, "a");
            }
            for (Cell c : this) {
                c.Draw();
            }
        }

        // Run the simulation
        int sizeAtProgression = (int) (1.2*Util.ArraySum(initPopSize));
        int refSize = Util.ArraySum(initPopSize); // Reference population size used for AT
        if (initialConditionType.equalsIgnoreCase("circle")) { // In principle could merge with the above but decided not to so to accidentally introduce error that might affect all previous results
            sizeAtProgression = (int) (1.2*Util.ArraySum(cellCountsArr));
            refSize = Util.ArraySum(cellCountsArr);
        }
        double currIntervalEnd;
        int currentPopSize = 0;
        int prevPopSize = 0;
        int nIntervals = (int) Math.ceil(tEnd/intervalLength);
        double prevDrugConcentration = currDrugConcentration;
        Boolean logged = false;
        for (int intervalIdx=0; intervalIdx<nIntervals+1;intervalIdx++) {
            currIntervalEnd = (tIdx-1)*dt+intervalLength;
            completedSimulationB = false;
            // Simulate the current interval
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                completedSimulationB = tIdx*dt>currIntervalEnd;
            }
            // Check if one of the other stopping conditions was met
            prevPopSize = currentPopSize;
            currentPopSize = Util.ArraySum(cellCountsArr);
            if ((prevPopSize>sizeAtProgression && currentPopSize>sizeAtProgression && terminateAtProgression) | currentPopSize<1) {break;}
            // Update the drug concentration
            if (currentPopSize>refSize) {currDrugConcentration = 1;}
            else if (currentPopSize<(1-atThreshold)*refSize) {currDrugConcentration = 0;}
            else {currDrugConcentration = currDrugConcentration>0?1:0;}
            // Count the number of cycles
            if (prevDrugConcentration==0 && currDrugConcentration==1) {nCycles++;}
            prevDrugConcentration = currDrugConcentration;
        }

        // Close the simulation
//        currVis.Close();
        this.Close(logged);
    }

    // ------------------------------------------------------------------------------------------------------------
    public void PredictIntermittentSchedule(double initialPSALevel, int weeksOnTreatment, int[] previousFourMeasurements, Boolean simulateCT) {
        // Initialise visualisation window
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        Boolean logged = false;

        // Simulate using the pre-defined schedule up to-date
        this.Run();

        // Predict forward
        int currentPopSize = Util.ArraySum(cellCountsArr);;
        Boolean endOfTrial = false;
        Boolean progression;
        double normalPSALevel = 4/initialPSALevel*Util.ArraySum(initPopSize);
        double elevatedPSALevel = 10/initialPSALevel*Util.ArraySum(initPopSize);
        int waitTimeToNextFollowUp = 28; // 4 weeks between follow ups
        int[] previousThreeChanges = new int[] {0, 0, 0};
        currDrugConcentration = 1;
        while (endOfTrial==false) {
//            System.out.println("Time: "+tIdx*dt+", Current Population Size: "+currentPopSize+", Normal PSA Size: "+normalPSALevel+", Elevated PSA Size: "+elevatedPSALevel);

            // Test if patient should be put on treatment
            if (currentPopSize>elevatedPSALevel && currDrugConcentration==0) {
                currDrugConcentration = 1;
            }

            // Test whether patient can be taken off treatment
//            if (weeksOnTreatment==24 && currentPopSize >= normalPSALevel) {endOfTrial = true;}
//            if (weeksOnTreatment==32 && currentPopSize >= normalPSALevel) {endOfTrial = true;}
//            if (weeksOnTreatment==36) {currDrugConcentration=0; weeksOnTreatment=0;}
            if (weeksOnTreatment>=36 && currentPopSize <= normalPSALevel && simulateCT==false) {currDrugConcentration=0; weeksOnTreatment=0;}

            // Test whether patient has progressed
            if (currDrugConcentration>0 && currentPopSize>normalPSALevel) {
                for (int i=0; i<3; i++) {previousFourMeasurements[i] = previousFourMeasurements[i+1];}
                previousFourMeasurements[3] = currentPopSize;
                for (int i=0; i<3; i++) {previousThreeChanges[i] = previousFourMeasurements[i+1] - previousFourMeasurements[i];}
                progression = true;
                for (int i=0; i<3; i++) {if (previousThreeChanges[i]<=0) {progression = false; break;}}
                endOfTrial = endOfTrial || progression;
            } else {previousFourMeasurements = new int[] {0, 0, 0, 0}; previousThreeChanges = new int[] {0, 0, 0};}

            // Break criterion to avoid indefinite loops
            if (tIdx*dt>3.65e3) {endOfTrial=true;}

            if (endOfTrial) {break;}
            this.SetTreatmentSchedule(new double[][]{{tIdx*dt,tIdx*dt+waitTimeToNextFollowUp,currDrugConcentration}});
            this.Run();
            weeksOnTreatment += currDrugConcentration>0? 4:0;
            currentPopSize = Util.ArraySum(cellCountsArr);
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    public void ProfileProProlifAT(double atThreshold, double intervalLength, double proProlifFactor) {
        ProfileProProlifAT(atThreshold, intervalLength, proProlifFactor,  1.);
    }

    public void ProfileProProlifAT(double atThreshold, double intervalLength, double proProlifFactor, double relEffectOnR) {
        // Initialise visualisation window
//        GridWindow currVis = new GridWindow(xDim,yDim,scaleFactor,false,null, visualiseB);//used for visualization
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        currDrugConcentration = 1.;

        // Set up the grid and initialise log if this is the beginning of the simulation
        if (tIdx==0) {
            if (initialConditionType=="random") {
                InitSimulation_Random(initPopSize[0],initPopSize[1]);
            } else if (initialConditionType=="separation1d") {
                InitSimulation_Separation1d(dist, initPopSize[0]);
            } else if (initialConditionType=="separation2d") {
                InitSimulation_Separation2d(dist,initPopSize[0]);
            } else if (initialConditionType=="circle") {
                InitSimulation_Circle(initRadius,initPopSize[1]/(initPopSize[0]+initPopSize[1]));
            }
            PrintStatus(0);
            if (cellCountLogFile==null && cellCountLogFileName!=null) {InitialiseCellLog(this.cellCountLogFileName);}
            SaveCurrentCellCount(0);
            SaveTumourImage(tIdx);
            tIdx = 1;
        } else {
            // Continue from a restart
            if (cellCountLogFileName!=null) {
                cellCountLogFile = new FileIO(cellCountLogFileName, "a");
            }
            for (Cell c : this) {
                c.Draw();
            }
        }

        // Run the simulation
        int sizeAtProgression = (int) (1.2*Util.ArraySum(initPopSize));
        double currIntervalEnd;
        int currentPopSize = 0;
        int prevPopSize = 0;
        int refSize = Util.ArraySum(initPopSize); // Reference population size used for AT
        int nIntervals = (int) Math.ceil(tEnd/intervalLength);
        double prevDrugConcentration = currDrugConcentration;
        double[] divRateArr;
        double[] normalGrowthRates = new double[] {divisionRate_S,divisionRate_R};
        double[] proProlifGrowthRates = new double[] {(1+proProlifFactor)*normalGrowthRates[0],(1+proProlifFactor*relEffectOnR)*normalGrowthRates[1]};
        Boolean logged = false;
        for (int intervalIdx=0; intervalIdx<nIntervals+1;intervalIdx++) {
            currIntervalEnd = (tIdx-1)*dt+intervalLength;
            completedSimulationB = false;
            // During the off-drug phase we will simulate administration of a pro-proliferation drug
            // by increasing the growth rates of the cells.
            divRateArr = currDrugConcentration==0? proProlifGrowthRates:normalGrowthRates;
            for (Cell c : this) {
                c.divisionRate=divRateArr[c.resistance];
            }

            // Simulate the current interval
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                completedSimulationB = tIdx*dt>currIntervalEnd;
            }
            // Check if one of the other stopping conditions was met
            prevPopSize = currentPopSize;
            currentPopSize = Util.ArraySum(cellCountsArr);
            if ((prevPopSize>sizeAtProgression && currentPopSize>sizeAtProgression && terminateAtProgression) | currentPopSize<1) {break;}
            // Update the drug concentration
            if (currentPopSize>refSize) {currDrugConcentration = 1;}
            else if (currentPopSize<(1-atThreshold)*refSize) {currDrugConcentration = 0;}
            else {currDrugConcentration = currDrugConcentration>0?1:0;}
            // Count the number of cycles
            if (prevDrugConcentration==0 && currDrugConcentration==1) {nCycles++;}
            prevDrugConcentration = currDrugConcentration;
        }

        // Close the simulation
//        currVis.Close();
        this.Close(logged);
    }

    // ------------------------------------------------------------------------------------------------------------
    public void ProfileProTurnoverAT(double atThreshold, double intervalLength, double proTurnoverFactor, double relEffectOnR) {
        // Initialise visualisation window
//        GridWindow currVis = new GridWindow(xDim,yDim,scaleFactor,false,null, visualiseB);//used for visualization
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        currDrugConcentration = 1.;

        // Set up the grid and initialise log if this is the beginning of the simulation
        if (tIdx==0) {
            if (initialConditionType=="random") {
                InitSimulation_Random(initPopSize[0],initPopSize[1]);
            } else if (initialConditionType=="separation1d") {
                InitSimulation_Separation1d(dist, initPopSize[0]);
            } else if (initialConditionType=="separation2d") {
                InitSimulation_Separation2d(dist,initPopSize[0]);
            } else if (initialConditionType=="circle") {
                InitSimulation_Circle(initRadius,initPopSize[1]/(initPopSize[0]+initPopSize[1]));
            }
            PrintStatus(0);
            if (cellCountLogFile==null && cellCountLogFileName!=null) {InitialiseCellLog(this.cellCountLogFileName);}
            SaveCurrentCellCount(0);
            SaveTumourImage(tIdx);
            tIdx = 1;
        } else {
            // Continue from a restart
            if (cellCountLogFileName!=null) {
                cellCountLogFile = new FileIO(cellCountLogFileName, "a");
            }
            for (Cell c : this) {
                c.Draw();
            }
        }

        // Run the simulation
        int sizeAtProgression = (int) (1.2*Util.ArraySum(initPopSize));
        double currIntervalEnd;
        int currentPopSize = 0;
        int prevPopSize = 0;
        int refSize = Util.ArraySum(initPopSize); // Reference population size used for AT
        int nIntervals = (int) Math.ceil(tEnd/intervalLength);
        double prevDrugConcentration = currDrugConcentration;
        double[] deathRateArr;
        Boolean logged = false;
        double[] normalDeathRates = new double[] {deathRate_S,deathRate_R};
        // Check if turnover is 0. If so, then we will add turnover.
        double[] proTurnoverRates;
        if (deathRate_S==0) {
            proTurnoverRates = new double[] {proTurnoverFactor*divisionRate_S,proTurnoverFactor*relEffectOnR*divisionRate_R};
        } else {
            proTurnoverRates = new double[] {(1+proTurnoverFactor)*normalDeathRates[0],(1+proTurnoverFactor*relEffectOnR)*normalDeathRates[1]};
        }

        for (int intervalIdx=0; intervalIdx<nIntervals+1;intervalIdx++) {
            currIntervalEnd = (tIdx-1)*dt+intervalLength;
            completedSimulationB = false;
            // During the off-drug phase we will simulate administration of a low dose chemo
            // by increasing the death rates of the cells.
            deathRateArr = currDrugConcentration==0? proTurnoverRates:normalDeathRates;
            for (Cell c : this) {
                c.deathRate=deathRateArr[c.resistance];
            }

            // Simulate the current interval
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                completedSimulationB = tIdx*dt>currIntervalEnd;
            }
            // Check if one of the other stopping conditions was met
            prevPopSize = currentPopSize;
            currentPopSize = Util.ArraySum(cellCountsArr);
            if ((prevPopSize>sizeAtProgression && currentPopSize>sizeAtProgression && terminateAtProgression) | currentPopSize<1) {break;}
            // Update the drug concentration
            if (currentPopSize>refSize) {currDrugConcentration = 1;}
            else if (currentPopSize<(1-atThreshold)*refSize) {currDrugConcentration = 0;}
            else {currDrugConcentration = currDrugConcentration>0?1:0;}
            // Count the number of cycles
            if (prevDrugConcentration==0 && currDrugConcentration==1) {nCycles++;}
            prevDrugConcentration = currDrugConcentration;
        }

        // Close the simulation
//        currVis.Close();
        this.Close(logged);
    }

    // ------------------------------------------------------------------------------------------------------------
    public void RunProTurnoverControl(double[][] treatmentScheduleList, double proTurnoverFactor, double relEffectOnR) {
        // Initialise visualisation window
        UIGrid currVis = new UIGrid(xDim,yDim,scaleFactor,visualiseB);
        this.vis = currVis;
        Boolean completedSimulationB = false;
        currDrugConcentration = 1.;

        // Set up the grid and initialise log if this is the beginning of the simulation
        if (tIdx==0) {
            if (initialConditionType=="random") {
                InitSimulation_Random(initPopSize[0],initPopSize[1]);
            } else if (initialConditionType=="separation1d") {
                InitSimulation_Separation1d(dist, initPopSize[0]);
            } else if (initialConditionType=="separation2d") {
                InitSimulation_Separation2d(dist,initPopSize[0]);
            } else if (initialConditionType=="circle") {
                InitSimulation_Circle(initRadius,initPopSize[1]/(initPopSize[0]+initPopSize[1]));
            }
            PrintStatus(0);
            if (cellCountLogFile==null && cellCountLogFileName!=null) {InitialiseCellLog(this.cellCountLogFileName);}
            SaveCurrentCellCount(0);
            SaveTumourImage(tIdx);
            tIdx = 1;
        } else {
            // Continue from a restart
            if (cellCountLogFileName!=null) {
                cellCountLogFile = new FileIO(cellCountLogFileName, "a");
            }
            for (Cell c : this) {
                c.Draw();
            }
        }

        // Run the simulation
        int currentPopulationSize = Util.ArraySum(initPopSize);
        int sizeAtProgression = (int) (1.2*currentPopulationSize);
        Boolean logged = false;
        double[] deathRateArr;
        Boolean secondaryDrugOn;
        double[] normalDeathRates = new double[] {deathRate_S,deathRate_R};
        double[] proTurnoverRates;
        // Check if turnover is 0. If so, then we will add turnover.
        if (deathRate_S==0) {
            proTurnoverRates = new double[] {proTurnoverFactor*divisionRate_S,proTurnoverFactor*relEffectOnR*divisionRate_R};
        } else {
            proTurnoverRates = new double[] {(1+proTurnoverFactor)*normalDeathRates[0],(1+proTurnoverFactor*relEffectOnR)*normalDeathRates[1]};
        }
        double currIntervalEnd;
        for (int intervalIdx=0; intervalIdx<treatmentScheduleList.length;intervalIdx++) {
            currIntervalEnd = treatmentScheduleList[intervalIdx][1];
            nTSteps = (int) Math.ceil(currIntervalEnd/dt);
            completedSimulationB = false;
            // During the off-drug phase we will simulate administration of a low dose chemo
            // by increasing the death rates of the cells.
//            currDrugConcentration = treatmentScheduleList[intervalIdx][2];
            secondaryDrugOn = treatmentScheduleList[intervalIdx][2]==0;
            deathRateArr = secondaryDrugOn? proTurnoverRates:normalDeathRates;
            for (Cell c : this) {
                c.deathRate=deathRateArr[c.resistance];
            }
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                completedSimulationB = tIdx*dt>=currIntervalEnd;
            }
        }
        // Check that the tumour has progressed by the end of the adaptive protocol. If it hasn't,
        // continue until progression
        currentPopulationSize = Util.ArraySum(cellCountsArr);
        if (currentPopulationSize<sizeAtProgression) {
            for (Cell c : this) {c.deathRate=normalDeathRates[c.resistance];}
            completedSimulationB = false;
            while (!completedSimulationB) {
                vis.TickPause(pause);
                StepCells();
                PrintStatus(tIdx);
                logged = SaveCurrentCellCount(tIdx);
                SaveTumourImage(tIdx);
                tIdx++;
                currentPopulationSize = Util.ArraySum(cellCountsArr) ;
                completedSimulationB = (currentPopulationSize>sizeAtProgression) | (currentPopulationSize<1) | (tIdx*dt>tEnd);
            }
        }

        // Close the simulation
        this.Close(logged);
    }

    // ------------------------------------------------------------------------------------------------------------
    // Manage and save output
    public void InitialiseCellLog(String cellCountLogFileName) {
        InitialiseCellLog(cellCountLogFileName, 1.);
    }

    public void InitialiseCellLog(String cellCountLogFileName, double frequency) {
        InitialiseCellLog(cellCountLogFileName,frequency,false);
    }

    public void InitialiseCellLog(String cellCountLogFileName, double frequency, Boolean profilingMode) {
        cellCountLogFile = new FileIO(cellCountLogFileName, "w");
        WriteLogFileHeader();
        this.cellCountLogFileName = cellCountLogFileName;
        this.logCellCountFrequency = frequency;
        if (profilingMode) {
            this.profilingMode = profilingMode;
            double[] tmpArr = GetModelState();
            int extraFields = extraSimulationInfoNames==null? 0: extraSimulationInfoNames.length;
            this.outputArr = new double[5][tmpArr.length+extraFields];
            // Initialise the logging array
            for (int i=0; i<outputArr.length; i++) {for (int j=0; j<outputArr[0].length; j++) {outputArr[i][j] = 0;}}
        }
    }

    private void WriteLogFileHeader() {
        cellCountLogFile.Write("TIdx,Time,NCells_S,NCells_R,NCells,DrugConcentration,rS,rR,mS,mR,dS,dR,dD,dt,NCycles,NAttemptedDivs,NFailedDivs,NDeaths");
        if (extraSimulationInfoNames!=null) {
            cellCountLogFile.Write(",");
            cellCountLogFile.WriteDelimit(extraSimulationInfoNames, ",");
        }
        cellCountLogFile.Write("\n");
    }

    public void SetExtraSimulationInfo(String[] extraInfoNamesArr, double[] extraInfoArr) {
        this.extraSimulationInfoNames = extraInfoNamesArr;
        this.extraSimulationInfo = extraInfoArr;
    }

    public Boolean SaveCurrentCellCount(int currTimeIdx) {
        Boolean successfulLog = false;
        if ((currTimeIdx % (int) (logCellCountFrequency/dt)) == 0 && logCellCountFrequency > 0) {
            if (!profilingMode) {
                cellCountLogFile.WriteDelimit(GetModelState(),",");
                if (extraSimulationInfoNames!=null) {
                    cellCountLogFile.Write(",");
                    cellCountLogFile.WriteDelimit(extraSimulationInfo, ",");
                }
                cellCountLogFile.Write("\n");
                successfulLog = true;
            } else {
                // Make space for new entry in data array
                for (int j=0; j<outputArr[0].length; j++) {
                    for (int i = outputArr.length-1; i > 0; i--) {
                        outputArr[i][j] = outputArr[i-1][j];
                    }
                }

                // Log the new entry
                double[] currModelState = GetModelState();
                for (int j=0; j<currModelState.length; j++) {
                    outputArr[0][j] = currModelState[j];
                }
                if (extraSimulationInfoNames!=null) {
                    for (int j=0; j<extraSimulationInfo.length; j++) {
                        outputArr[0][currModelState.length+j] = extraSimulationInfo[j];
                    }
                }
                successfulLog = true;
            }
        }
        return successfulLog;
    }

    public void PrintStatus(int currTimeIdx) {
        if (verboseLevel > 0 && (currTimeIdx % (int) (printFrequency/dt)) == 0) {
            System.out.println("Time: " + currTimeIdx*dt+
                    " - Population Size: " + Util.ArraySum(cellCountsArr)+
                    " - Drug Concentration: " + currDrugConcentration);
        }
    }

    public void SaveTumourImage(int currTimeIdx) {
        if (imageFrequency > 0 && (currTimeIdx % (int) (imageFrequency/dt)) == 0) {
            this.vis.ToPNG(imageOutDir +"/img_t_"+currTimeIdx*dt+".png");
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // Function to clean up loose ends after running a simulation.
    public void Close() {
        if (cellCountLogFile!=null) {cellCountLogFile.Close();}
    }

    public void Close(Boolean logged) {
        if (!logged) {
            tIdx--;
            SaveCurrentCellCount(0);}
        if (profilingMode) {
            for (int i = 0; i < outputArr.length; i++) {
                cellCountLogFile.WriteDelimit(outputArr[i],",");
                cellCountLogFile.Write("\n");
            }
        }
        if (cellCountLogFile!=null) {cellCountLogFile.Close();}
    }
}