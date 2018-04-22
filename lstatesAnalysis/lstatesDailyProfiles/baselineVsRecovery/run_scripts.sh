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
BASE_DIR="/home/vkatsageorgiou/Documents/experiments/miceSleep/testCode/sci/"

# training epochID
epochID=9999

# degrees of polynomial, i.e. deg = 8 for cd1c57mixed, Zfhx3
deg=8


if [ ! -f "${BASE_DIR}doneHomoResp" ] 
	then
		echo "computing homostatic response"
		
		python2 -W ignore baselineRecoveryResponsePerStage.py -refDir ${BASE_DIR} -epoch ${epochID} -deg ${deg}
else
    echo "computing the max absolute homostatic response per latent state"
    python2 -W ignore baselineRecoveryResponseMaxAbsDifference.py -refDir ${BASE_DIR} -epoch ${epochID} -deg ${deg}
fi
