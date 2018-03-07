THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE FILE
                AT THE SOURCE DIRECTORY.

Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved

@author : vasiliki.katsageorgiou@gmail.com


PUBLICATION : A Novel Unsupervised Analysis of Electrophysiological Signals 
                        Reveals New Sleep Sub-stages in Mice

The given scripts perfom unsupervised analysis of EEG/EMG mouse data employing 
the mean-covariance Restricted Boltzmann Machine (mcRBM) by Marc'Aurelio Ranzato.

Refer to:
"M. Ranzato, G. Hinton, "Modeling Pixel Means and Covariances Using Factorized 
Third-Order Boltzmann Machines", CVPR 2010"


REQUIREMENTS:
            - an NVIDIA GPU
            - NVIDIA drivers
            - cuda
            - python 2.7

Python packages needed:
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

            
The code has been developed and tested on arch linux.

********************************************************************************

HOW TO RUN THE CODE:

Each sub-directory includes a bash script that can be run in order to run the 
corresponding scripts.

To perform the analysis presented in the paper, follow the next steps:

Step 1: model training
    - create a folder where the analysis will be stored & copy inside the 
      configuration files that can be found at ./trainModel/configurationFiles/
      Modify the configuration files accordingly
    - go to ./trainModel/ & run the "run_gpu.sh" script after having modifying it
      accordingly
      
Step 2: latent states inference
    - the latent states of a trained mcRBM can be inferred using the "runScripts.sh"
      that can be found inside ./lstatesInference/
      
      NOTE that every 100 training epochs, the weights of model are being stored and
      can be used to perform the inference of the latent states as well as the further
      analysis steps even before the predefined training epochs have been reached.
      
Step 3: analysis of the inferred latent states
    - modify accordingly & run the "runScripts.sh"
      (see in lstatesAnalysis.py for more details regaring the output of the script)
      
    - go to ./lstatesDailyProfiles/ and run the "runScripts.sh" to get the daily profiles
      of the inferred latent states.
      
Step 4: mouse strains classification
      
    In case of multi-subject where different mouse strains have been included, you can 
    whether you can discriminate among mouse groups using the classification approach 
    that has been implemented in the scripts found in ./discriminateMouseGroups/
    
    To do so, modify & run the "runScripts.sh".
    
