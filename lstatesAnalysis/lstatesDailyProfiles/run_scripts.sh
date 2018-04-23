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
BASE_DIR="/home/vkatsageorgiou/mouseSleepAnalysis/experiment2/"

# training epochID
epochID=9999

# degrees of polynomial, i.e. deg = 8 for cd1c57mixed, Zfhx3
deg=8

# strains: "cd1c57mixed" OR "Zfhx3"
strains="Zfhx3"


if [ ! -f "${BASE_DIR}doneDailyProfiles" ] 
	then
		echo "1: computing daily profiles"
		
		python2 -W ignore lstatesDailyProfiles.py -f ${BASE_DIR} -epoch ${epochID} -case ${strains} -deg ${deg}
else
    echo "computing the per latent state daily profile's peaks"
    python2 -W ignore peaksDetection.py -f ${BASE_DIR} -epoch ${epochID} -case ${strains} -deg ${deg}
fi

if [ -f "${BASE_DIR}doneDailyProfiles" ] 
	then
		echo "2: computing the max absolute homostatic response per latent state"
		python2 -W ignore peaksDetection.py -f ${BASE_DIR} -epoch ${epochID} -case ${strains} -deg ${deg}
fi
