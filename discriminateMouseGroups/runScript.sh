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
refDir='/home/vkatsageorgiou/documents/experiments/mouseSleepTucci/sci/baseline2recovery/analysis/epoch4500/'

normFlag=0

python2 classificationEsnemble.py -refDir ${refDir} -normFlag ${normFlag}
