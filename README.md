# Strobl et al (xxx). *Spatial Structure Impacts Adaptive Therapy by Shaping Intra-Tumoral Competition*
This repository contains the code and data for our publication Strobl et al (xxx). *Spatial Structure Impacts Adaptive Therapy by Shaping Intra-Tumoral Competition*, which is currently under review. A pre-print of our manuscript is available on the [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.11.03.365163v2) [1].

![Gif showing simulations of our tumour model under continuous and adaptive therapy. Continuous therapy results in rapid competitive release, whereas adaptive therapy is able to maintain control of the resistant colonies for longer.](gifs/supplementaryMovie1.gif)

## Requirements
A full list of the Python packages used in this project can be found in `requirements.txt`. To recreate the virtual environment, run:
```console
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r requirements.txt
``` 
For further details, see [here](https://stackoverflow.com/questions/14684968/how-to-export-virtualenv)

In addition, in order to create the neighbourhood plots (e.g. Figure 2d), you will require EvoFreq in R [2]. See [the Github page](https://github.com/MathOnco/EvoFreq) for further instructions.

## Computational model
The model is implemented in Java 1.8. in [HAL](https://halloworld.org/index.html) [3]. The code can be found inside the `abm/onLatticeCA` directory:
- `OnLatticeCA.java` contains the main model class and simulation functions.
- `Cell.java` contains the class used to model individual cells in the model.
- `runParameterSweep.java` houses a wrapper that allows to run simulations from the command line. All data presented in the paper was collected using this wrapper.

All other files and directories inside `abm/onLatticeCA` are part of HAL, and were downloaded in their current form from [HAL's Github repository](https://github.com/MathOnco/HAL). There are different ways to run the model. We recommend using the compiled  `onLatticeModel.jar` executable, which was how we collected all the data presented in the manuscript. In fact, if you look at the jupyter notebooks (e.g. `jnb_figure2.ipynb`) you can see how you can run it directly from jupyter. Alternatively, you can run it from the command line by calling your java VM. An example of that may look as follows:
```console
java -jar onLatticeModel.jar -initialSize 0.75 -rFrac 0.001 -turnover 0 -cost 0 -tEnd 3650 -seed 0 -nReplicates 1 -profilingMode false -terminateAtProgression true -imageOutDir ./data/exampleSims_noCost_noTurnover/images/ -imageFreq 10 -outDir ./data/scratch/
```

Finally, you can also compile and run the java code yourself. For instructions of how to do so, see the HAL manual (`abm/manual.pdf`).

## Analysis
For each results figure in the manuscript we have created a separate jupyter notebook which houses the code to re-create this figure. These are named `jnb_figure2.ipynb` etc. 

## References
- ﻿[1] Strobl, M. A. R., Gallaher, J., West, J., Robertson-Tessi, M., Maini, P. K., & Anderson, A. R. A. (2020). Spatial structure impacts adaptive therapy by shaping intra-tumoral competition. BioRxiv, 2020.11.03(365163), doi: https://doi.org/10.1101/2020.11.03.365163.
- ﻿[2] Gatenbee, C. D., Schenck, R. O., Bravo, R. R., & Anderson, A. R. A. (2019). EvoFreq: visualization of the Evolutionary Frequencies of sequence and model data. BMC Bioinformatics, 20(1), 710. https://doi.org/10.1186/s12859-019-3173-y
- ﻿[3] Bravo, R. R., Baratchart, E., West, J., Schenck, R. O., Miller, A. K., Gallaher, J., … Anderson, A. R. A. (2020). Hybrid Automata Library: A flexible platform for hybrid modeling with real-time visualization. PLoS Computational Biology, 16(3), e1007635. https://doi.org/10.1371/journal.pcbi.1007635
