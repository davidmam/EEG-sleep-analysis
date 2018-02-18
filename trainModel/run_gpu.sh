#
#   THIS CODE IS UNDER THE MIT LICENSE. YOU CAN FIND THE COMPLETE FILE
#   				AT THE SOURCE DIRECTORY.
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
BASE_DIR="/run/media/vkatsageorgiou/data/mouseSleepAnalysis/journalpaper_dsetsAsWebpage/experiment1/"

if [ ! -f "${BASE_DIR}done" ] 
	then
		echo "starting ${BASE_DIR}"
		python2 trainModel.py -f "${BASE_DIR}" -gpuId 0
		#echo "letting gpu 0 cool down" #-- Uncomment in case you want to run multiple experiments
		#sleep 1800  #-- Uncomment in case you want to run multiple experiments
fi
