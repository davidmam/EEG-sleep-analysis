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

# in case of multi-subject analysis,
# provide the mouse groups names (ex. Mixed,CD1,C57BL/6J  or Zfhx3Sci/+,Zfhx3+/+)
# using the order subjects have been concatenated in the dataset
groupNames=Zfhx3Sci/+,Zfhx3+/+

epochID=9999

# latent states smaller than the threshold will be removed for the visualization of the heatMap
# and the computation of the entropy and mutual information
threshold=1

# specify if multiple-subject analysis is performed or not as
# multi: "true" or "false"
multi=true

# specify which normalization to be used for the scaling of the histogram over the latent states, like:
# norm : "L2" or "other"
norm=L2

# specify which features were used for the analysis, like:
# features : "bands" or "ratios" or "other"
features=ratios

python2 lstatesAnalysis.py -f ${BASE_DIR} -epochID ${epochID} -threshold ${threshold} -multi ${multi} -norm ${norm} -features ${features} -groupNames ${groupNames}
