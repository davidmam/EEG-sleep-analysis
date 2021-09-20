#
#   THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
#                    FILE AT THE SOURCE DIRECTORY.
#                
#       Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
#    
#               @author : vasiliki.katsageorgiou@gmail.com
#
#    
#                        Publication:
#    A Novel Unsupervised Analysis of Electrophysiological
#        Signals Reveals New Sleep Sub-stages in Mice
#        
#        
#*****************************************************************************



#!/bin/bash
BASE_DIR="/home/vkatsageorgiou/mouseSleepAnalysis/experiment1/"

# training epochID
epochID=9999

# degrees of polynomial, i.e. deg = 8 for cd1c57mixed, Zfhx3
deg=8

# strains: "cd1c57mixed" OR "Zfhx3"
strains="cd1c57mixed"


if [ -f "${BASE_DIR}doneDailyProfilesGT" ] 
    then
        echo "3: computing the per manually scored sleep stage daily profiles"
        python -W ignore groundTruthStageProfiles.py -f ${BASE_DIR} -epoch ${epochID} -case ${strains} -deg ${deg}
fi
