#
#   THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
#					FILE AT THE SOURCE DIRECTORY.
#				
#       Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
#	
#           	@author : vasiliki.katsageorgiou@gmail.com
#
#	
#						Publication:
#	A Novel Unsupervised Analysis of Electrophysiological
#		Signals Reveals New Sleep Sub-stages in Mice
#		
#		
#*****************************************************************************


#!/bin/bash

# this script is call the trainModel.py script that is responsible for the
# training of the mcRBM

# Set path to the directory where the analysis will be stored:
# NOTE that the configuration files that can be found in the
# "configurationFiles" folder need to have been stored in the same directory.
BASE_DIR="/home/vkatsageorgiou/mouseSleepAnalysis/experiment1/"

if [ ! -f "${BASE_DIR}done" ] 
	then
		echo "starting ${BASE_DIR}"
		
		# set the id of the GPU to be used for the computations
		python2 trainModel.py -f "${BASE_DIR}" -gpuId 0
		
		#echo "letting gpu 0 cool down" #-- Uncomment in case you want to run multiple experiments
		#sleep 1800  #-- Uncomment in case you want to run multiple experiments
fi
