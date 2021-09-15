# Tumour Growth Patterns Impact Evolution

## Agent-based models reveal the impact of growth patterns on spatial and temporal features of clonal diversification

*Nature Ecology & Evolution* (Provisionally accepted) ; *Research Square* ([Preprint Link](https://www.researchsquare.com/article/rs-244285/v1 "Preprint on Research Square"))

**Summary of the study**: In this study, we developed an agent-based model of tumour growth and clonal evolution to study the features of clonal diversification in space and time. We linked our modelling analysis to the multi-region sequencing data in the Tracking Renal Cell Cancer Evolution through therapy (TRACERx Renal) study. Through computational modelling, we found that distinct growth patterns, specifically, Surface Growth and Volume Growth, give rise to different extents and spatio-temporal features of clonal diversification. In corraboration with data, "power law" patterns characterising the spatial features of clonal diveristy in the model are also observed in the ccRCC tumours and, interestingly, show an association with the rate of progression according to published clinical annotation. Overall, Surface Growth models reflect more branched tumours with attenuated progression, while Volume Growth models stochastically lead to dichotomous patterns of evolution and reflect either indolent tumours with lack of evolution or aggressive tumours with rapid progression. *In-silico* time-course studies reveal divergent temporal trajectories of evolution in Surface and Volume Growth models, which plausibly explain the apparent non-monotic relationship between clonal diversity and primary tumour sizes in ccRCCs. A subset of early-stage tumours with radiological evidence of budding structures, which are early indicators of subclonal advantageous outgrowth in Surface Growth models, are predicted to undergo more extensive clonal diversification. 


## About this repository
This repository contains CUDA C++ Source Code developed for the study as well as Source Data and Scripts for producing Main Figures and Extended Data Figures of the paper.

## Model and Source Code

### Model
A coarse-grained cellular automaton model is developed for this study. See the paper link above for more details.

### Source Code
The computer code is written in CUDA C++. A brief description of functions and key parameters in the code, output files produced, and compilation of the code is given below.

#### functions
* `main(...)` function: control of simulation flow, including intialisation, iterations of growth, death, and driver acquisition, and writing outputs.
* `growth_random_kernel(...)`, `growth2(...)`, and `growth2_volume(...)` functions: implement death and growth events
* `emerge_subclones_rcc_uponProlif(...)` function: implement acquisition of driver events and accordingly the emergence of subclones
* `necrosis_kernel(...)` and `necrosis2(...)` functions: implement central necrosis
* `update_surface_kernel(...)` and `update_surface2(...)` functions: get updated surface voxels
* `writeDynamics(...)` function: write output files

#### Key parameters
* `typeGrowthMode`: indicate whether to perform simulations with Surface Growth model ('s') or Volume Growth model ('v')
* `typeDriverAdvantage`: indicate whether to perform simulations with Saturated model ('s') or Additive model ('a') of driver advantages
* `flagSaveCellDynamicsOverTime`: boolean variable to indicate whether to save model snapshots over time, by default, false. This is set to true for time-course experiments.
* `flagTumourApop`: boolean variable to indicate whether to consider death events, by default, true. This is set to false for simulations with necrosis turned on.
* `flagTumourNecr`: boolean variable to indicate whether to turn on necrosis module, by default, false. This is set to true for simulations with necrosis turned on.
* `P_COPY`: the probability of tumour voxel duplication. (p_growth in the paper)
* `P_EMPTY`: the probability of tumour voxel death. (p_death in the paper)
* `P_EVENT_DRIVER_RCC_UPON_PROLIF`: the probability of driver acquisition. (p_driver in the paper)
* `P_NECROSIS`: the probability of death due to necrosis, when necrosis module is turned on. (p_necrosis in the paper)

#### Output files
* `*cellDynamics.txt`: positions and clone ids of tumour voxels on the 3D tumour surface. 
* `*cellDynamicsXY.txt`: positions and clone ids of tumour voxels within the 2D tumour section.
* `*cloneEventsOrder.txt`: ids of paried child and parent clones.
* `*cloneEvents.txt`: clone id and driver events specific to that clone.
* `*cloneSizeOverTime.txt`: prevalence (i.e., number of tumour voxels harbouring a given clone) of clones over time.
* `*eventSizeOverTime.txt`: prevalence (i.e., number of tumour voxels harbouring a given clone) of RCC drivers over time.

#### Compilation
The code can be complied by running command `cmake ../ && make -j` in a sub-directory (e.g., `src/build/`) of the directory (e.g., `src/`) that contains the source code. Cmake (version 3.12.1) and Cuda compilation tools (release 9.2) were used in a Linux environment in this study.
An exemplar folder structure is given below.
```
└── src
    ├── build
    ├── CMakeLists.txt
    ├── tumour_growth_patterns.cu
    └── tumour_growth_patterns.cuh
```
After compilation, an executable named `tumour_growth_patterns` is created. 

## Source Data and Scripts

### Source Data
Source Data include Excel workbooks containing data for Main Figures and Extended Data Figures presented in the paper, with figure number indicated in the name of workbooks.

### Scripts
Python or R scripts that read the Source Data for producing plots presented in the Main Figures and Extended Data Figures are provided for reference.

## License
Distributed under **The Francis Crick Institute License**.

## Contact
Please contact Xiao Fu and Paul A. Bates for questions about the source code.
* Xiao Fu - xiao.fu@crick.ac.uk or iamfuxiao@gmail.com; @XiaoFu90
* Paul A. Bates - paul.bates@crick.ac.uk; @PaulBatesBMM

## Acknowledgements
The code was developed in the collaboration between multiple research labs at the Francis Crick Institute:
* [Biomolecular Modelling Laboratory](https://www.crick.ac.uk/research/labs/paul-bates "Biomolecular Modelling")
* [Tumour Cell Biology Laboratory](https://www.crick.ac.uk/research/labs/erik-sahai "Tumour Cell Biology")
* [Cancer Dynamics Laboratory](https://www.crick.ac.uk/research/labs/samra-turajlic "Cancer Dynamics")
* [Cancer Evolution and Genome Instability Laboratory](https://www.crick.ac.uk/research/labs/charles-swanton "Cancer Evolution and Genome Instability")


