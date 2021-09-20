#
#   THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
#                      FILE AT THE SOURCE DIRECTORY.
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
refDir='/home/vkatsageorgiou/mouseSleepAnalysis/experiment1/'

# normFlag=1 : histograms are normalized using L2 normalization
# default: normFlag=0
normFlag=0

python -W ignore classificationEsnemble.py -refDir ${refDir} -normFlag ${normFlag}
