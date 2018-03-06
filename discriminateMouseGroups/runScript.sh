#
#   THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
#					  FILE AT THE SOURCE DIRECTORY.
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
refDir='/home/vkatsageorgiou/mouseSleepAnalysis/experiment1/'

normFlag=0

python2 -W ignore classificationEsnemble.py -refDir ${refDir} -normFlag ${normFlag}
