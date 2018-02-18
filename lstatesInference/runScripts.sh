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
BASE_DIR="/run/media/vkatsageorgiou/data/mouseSleepAnalysis/journalpaper_dsetsAsWebpage/experiment1/mcRBManalysis/"

modelName="ws_fac11_cov11_mean10.mat"

python2 inferStates.py -f ${BASE_DIR} -done "True" -m ${modelName}
