"""
	THIS CODE IS UNDER THE MIT LICENSE. YOU CAN FIND THE COMPLETE FILE
					AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************
		

Script for training the mean-covariance Restricted Boltzmann Machine (mcRBM)
to model electrophysiological data for the analysis of sleep in mice.

Run the script from a terminal as: 
		python2 trainModel.py -f "path_to_configuration_files" -gpuId X 

where 
	- "path_to_configuration_files" is a string with the path to the 
	   folder containing experiment-related data & configuration files
	- X is an integer number indicating the ID of the GPU to be used for 
	  computation

<vkatsageorgiou@vassia-PC>
"""

import os
import argparse
import numpy as np
from mcRBM import mcRBM

# parsing input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', help='folder containing experiment-related data')
parser.add_argument('-gpuId', help='ID of GPU to be used for computation')
args = parser.parse_args()

expFile = 'exp_details'
modelFile = 'input_configuration'

# the methods used below are documented in the class files of the object they belong to

#model = mcRBM(os.getcwd(), expFile, modelFile, args.gpuId)
model = mcRBM(args.f, expFile, modelFile, args.gpuId)

# loading data
model.loadData()

# trim for gpu
model.d, model.obsKeys, model.epochTime = model.dpp.trimForGPU(model.d, model.obsKeys, model.epochTime, model.batch_size)

np.savez(model.saveDir + '/initData.npz', data=model.d, obsKeys=model.obsKeys, epochTime=model.epochTime)

# further subsetting and scaling
#-- Pre-processing Data: preprocAndScaleData(self, d, obsKeys, logFlag, scalingFlag, scaling, pcaFlag, whitenFlag, rescalingFlag, minmaxFile, saveDir)
model.d, model.obsKeys, model.dMean, model.dStd, model.dMinRow, model.dMaxRow, model.dMin, model.dMax = \
	model.dpp.preprocAndScaleData(model.d, model.obsKeys, model.logFlag, model.meanSubtructionFlag, model.scaleFlag, model.scaling, model.doPCA, model.whitenFlag, model.rescaleFlag, model.rescaling, 'minmaxFileInit', model.saveDir)

# perform training
model.train()
