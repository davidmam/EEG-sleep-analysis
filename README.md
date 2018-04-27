## COPYRIGHT

THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE FILE
                        AT THE SOURCE DIRECTORY.

Copyright (C) 2017 V.-M. Katsageorgiou, D. Sona, V. Murino - All rights reserved

@author : vasiliki.katsageorgiou@gmail.com


This software was written as part of the research conducted at PAVIS, 
Istituto Italiano di Tecnognolia (IIT) - (https://pavis.iit.it/).


## PUBLICATION
V.-M. Katsageorgiou, D. Sona, M. Zanotto, G. Lassi, C. Garcia-Garcia, V. Tucci, V. Murino,  
"A Novel Unsupervised Analysis of Electrophysiological Signals Reveals New Sleep Sub-stages in Mice"

The given scripts perform unsupervised analysis of EEG/EMG mouse data employing  
the mean-covariance Restricted Boltzmann Machine (mcRBM) by Marc'Aurelio Ranzato.

Refer to:
"M. Ranzato, G. Hinton, "Modeling Pixel Means and Covariances Using Factorized 
Third-Order Boltzmann Machines", CVPR 2010"


## REQUIREMENTS
            - an NVIDIA GPU
            - NVIDIA drivers
            - cuda
            - python 2.7

## Python PACKAGES REQUIRED
            - cudamat
            - numpy
            - cPickle
            - matplotlib.pyplot
            - shutil
            - scipy.io
            - ConfigParser
            - argparse
            - sklearn
            - pickle
            - PIL.Image

            
## The code has been developed and tested on Arch Linux.

********************************************************************************

## HOW TO RUN THE CODE

Each sub-directory includes a bash script that can be run in order to run the 
corresponding scripts.

To perform the analysis presented in the paper, follow the next steps:

## Step 1: model training
    - create a folder where the analysis will be stored & copy inside the 
      configuration files that can be found at ./trainModel/configurationFiles/
      Modify the configuration files accordingly
	  
    - go to ./trainModel/ & run the "run_gpu.sh" script after having modified it
      accordingly
      
## Step 2: latent states inference
    - the latent states of a trained mcRBM can be inferred using the "runScripts.sh"
      that can be found inside ./lstatesInference/
      
      NOTE that every 100 training epochs, the weights of model are being stored and
      can be used to perform the inference of the latent states as well as the further
      analysis steps even before the predefined training epochs have been reached.
      
## Step 3: analysis of the inferred latent states
	- the script "lstatesAnalysis.py" that can be found in ./lstatesAnalysis/ produces
	  most of the results that have been summarized in the Figures of the paper.
	  Specifically, it produces the analysis & plots of the following Figures:
	  Fig 2, Fig 3, Fig 4 - top, boxplots of Fig 6, S1 Fig, S3 Fig.
	  
	  To run the analysis, modify accordingly & run the "runScripts.sh" that can be found 
	  in ./lstatesAnalysis/
      (see in lstatesAnalysis.py for more details regarding the output of the script)
      
    - go to ./lstatesDailyProfiles/ and run the "runScripts.sh" to get the daily profiles
      of the inferred latent states & to find their peaks.
	  
	  The script "lstatesDailyProfiles.py" computes and visualizes the per latent state
	  daily profiles, as shown in Figures: Fig 4 - bottom, Fig 6, S4 Fig.
	  
	  The script "peaksDetection.py" computes and visualizes the peaks in the daily profiles 
	  of the latent states. These results are shown in Fig 5.
	  
	- For comparison of the latent states' profiles to the ones of manually scored sleep stages,
	  use the scripts in ./groundTruthStageVisualization/
	  
	- In case of analysis of data including both baseline and recovery (after sleep deprivation) 
	  recordings, go to ./lstatesDailyProfiles/baselineVsRecovery/ to compute the homostatic
	  response of the observed latent states.
	  
	  The script "baselineRecoveryResponsePerStage.py" computes and visualizes the per 
	  latent state homostatic response, as shown in Fig 7 - C,D,E,F.
	  
	  The script "baselineRecoveryResponseMaxAbsDifference.py" computes and visualizes the maximum
	  absolute response of each latent state, results shown in Fig 7 - A,B.
      
## Step 4: mouse strains classification
      
    In case of multi-subject where different mouse strains have been included, you can 
    test whether you can discriminate among mouse groups using the classification approach 
    that has been implemented in the script that ca be found in "./discriminateMouseGroups/".
	
	The script "classificationEsnemble.py" produces the results shown in S2 Fig.
    
    To do so, modify & run the "runScripts.sh".
	
## Other Scripts
      
    The scripts found in ./groundTruthStageVisualization/ allow for comparison of the latent states
	to manually scored sleep stages. Specifically:
	  
	  - The script "GTStageBoxplots.py" allows for the comparison of the latent states' distributions
	   (boxplots in Fig 2, Fig 6 and S1 Fig) to the distributions of the manually scored sleep stages
	   (see S6 Fig, S7 Fig).
	  
	  - The script "groundTruthStageProfiles.py" allows for the comparison of the latent states' daily profiles
	   (curves in Fig 4, Fig 6 and S4 Fig) to those of the manually scored sleep stages (see Fig 6 -A, S5 Fig).
	   
## Datasets used 
      
    The code is also accompanied by the pre-processed data used in the experiments in this study.
	For more details, see README.md in ./datasetsPaper/.
    
## QUESTIONS
      
    For questions / possible problems, please contact:
	vasiliki.katsageorgiou@gmail.com

	