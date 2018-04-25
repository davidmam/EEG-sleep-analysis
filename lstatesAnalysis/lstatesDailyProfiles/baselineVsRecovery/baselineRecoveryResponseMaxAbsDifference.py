""" 
	THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
						FILE AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************
		

Script written for analysing the per strain daily profiles in the inferred 
(from a trained RBM model) latent states for the study of sleep in mice:

here we compute the maximum absolute homostatic response of the observed 
latent states in the recovery mode (i.e. after sleep deprivation) in respect
to the baseline.


*********************************  OUTPUT    *********************************

Output : a folder named as "homostaticResponse" including a graph visualizing 
		 the max absolute responses.
		 		
*****************************************************************************
		 		

<vkatsageorgiou@vassia-PC>
"""

from __future__ import division

import sys
import os

import numpy as np
from numpy.random import RandomState
from numpy.polynomial import Polynomial
from scipy.io import loadmat, savemat
from ConfigParser import *
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib import use
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from scipy.special import expit
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import stats
import cPickle
import shutil
from adjustText import adjust_text



class homostaticResponseAbsDiff(object):	
    '''
    Object for analyzing the daily profiles of the observed latent states,
    testing the response of the latent states in the Recovery mode
    (i.e. after sleep deprivation).
    It computes the max absolute difference of the response in each latent
    state.
	'''
	
    def __init__(self, refDir, epochID, polydeg):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # id of the epoch to be analysed
        self.epochID = int(epochID)
        # degree of polynomial
        self.polydeg = int(polydeg)
                
        np.random.seed(124)
        self.prng =  RandomState(123)
        
    def loadData(self):
		'''
		Method loading the visible data.
		'''
		visData = 'visData.npz'
		dataFile = np.load(self.refDir + visData)
		self.d = dataFile['data']
		self.epochIDs = dataFile['obsKeys'].astype(int)
		self.epochTime = dataFile['epochTime']#.astype(int)
		
		"""
		Move in final analysed model folder
		"""
		os.chdir(self.refDir + 'analysis/epoch%d' %self.epochID)
		
		"""
		Load analysed data: array with epochIDs & the corresponding
		latent-states' IDs.
		"""
		self.obsKeys = np.load('obsKeys.npz')['obsKeys'].astype(int)
		
		"""
		Associate each epoch with the strain ID
		"""
		self.obsKeys = np.insert(self.obsKeys, self.obsKeys.shape[1], 0, axis=1)
		for i in range(self.obsKeys.shape[0]):
			self.obsKeys[i, self.obsKeys.shape[1]-1] = int(str(self.obsKeys[i, self.obsKeys.shape[1]-2])[0])			
		
		self.lstates = np.unique(self.obsKeys[:, 1])
		self.subjIDs = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-2])
		self.strainIDs = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-1])
		
		#self.epochTime = self.epochTime[:self.obsKeys.shape[0],:]
		
		self.utc_transform()
    
    def utc_transform(self):
		'''
		Method for transforming the serial date-time to integer.
		'''
		
		# add the column with the mode id: 1-baseline, 2-recovery
		self.obsKeys = np.hstack((self.obsKeys, self.epochTime[:, self.epochTime.shape[1]-2].reshape(-1,1))).astype(int)
		
		self.epochTime = np.insert(self.epochTime, self.epochTime.shape[1], 0, axis=1)
		
		"""
		Iterate through epochs:
		"""
		for i in range(len(self.epochTime)):
			self.epochTime[i, self.epochTime.shape[1]-1] = int(datetime.datetime.utcfromtimestamp((self.epochTime[i, self.epochTime.shape[1]-2]-25569) * 86400.0).strftime("%H"))
		
		"""
		Create unique array: epochsIDs - lstatesIDs - subjects - mode - dayTime
		"""
		self.obsKeys = np.hstack((self.obsKeys, self.epochTime[:, self.epochTime.shape[1]-1].reshape(-1,1))).astype(int)
		#savemat('obsKeysTime.mat', mdict={'obsKeys':self.obsKeys})
	
    def combinedGroupHistograms(self):
		'''
		Method for visualizing the distribution of each latent state in time
		per group (strain).
		'''
		if not os.path.isdir('homostaticResponse'):
			os.makedirs('homostaticResponse')
					
		
		self.baselineHours = [13, 13, 13, 13, 13, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6]
		self.recoveryHours = [13, 13, 13, 13, 13, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6]
		self.baselineInd = np.arange(len(self.baselineHours))
		self.recoveryInd = np.arange(len(self.recoveryHours))
		
		
		plt.style.use('bmh')
		palete = plt.rcParams['axes.color_cycle']
		colors = ['#b2182b', palete[9], palete[8], '#1a1a1a', '#004529']
		
		
		fig, ax1 = plt.subplots(1, 1, figsize=(59,55))
							
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['left'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		
		ax1.grid(False)
		width = 0.6
		
		"""
		Iterate through latent states:
		"""
		data_x = []
		data_y = []
		labels = []
		dataStore = {}
		for si in self.strainIDs:
			dataStore['strain%d' %si] = np.array([])
		for self.lstate in self.lstates:
				
			idx = np.where(self.obsKeys[:,1]==self.lstate)[0]
			
			if len(idx) >= 1500:
				currObsKeys = self.obsKeys[idx,:]
				
				w = len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==1))[0])
				nr = len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==2))[0])
				r = len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==3))[0])
				
				epochsW = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==1))[0]) / len(currObsKeys)), 3)			
				epochsNR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==2))[0]) / len(currObsKeys)), 3)
				epochsR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==3))[0]) / len(currObsKeys)), 3)
				
				behaviorDistribution = [epochsW, epochsNR, epochsR]
				max_idx = behaviorDistribution.index(max(behaviorDistribution))
				
				if max_idx == 0:
					ci = 1
					sleep = 0
				elif (max_idx == 1 or max_idx == 2):
					ci = 0
					sleep = 1
				
				""" Iterate through strains	"""				
				maxStrainID = max(self.strainIDs)
				for self.strain in self.strainIDs:
					
					idxStr = np.where(currObsKeys[:, currObsKeys.shape[1]-3]==self.strain)[0]
					if len(idxStr) != 0:
						strainObsKeys = currObsKeys[idxStr, :]
						
						# spit into modes
						baseline = strainObsKeys[np.where( strainObsKeys[:, strainObsKeys.shape[1]-2] == 1 )[0], :]
						recovery = strainObsKeys[np.where( strainObsKeys[:, strainObsKeys.shape[1]-2] == 2 )[0], :]
						
						if (len(baseline) > 0. and len(recovery) > 0.):
									
							count_baseline, p_baseline = self.computeDistribution(baseline)
							count_recovery, p_recovery = self.computeDistribution(recovery)
							
							idx_b_start = np.where( p_baseline.linspace()[0] >= 6. )[0][0]
							idx_b_end = np.where( p_baseline.linspace()[0] >= len(self.baselineHours)-6 )[0][0]
							
							idx_r_start = np.where( p_recovery.linspace()[0] >= 6. )[0][0]
							idx_r_end = np.where( p_recovery.linspace()[0] >= len(self.recoveryHours)-6 )[0][0]
							
							"""
							Compute curve difference
							"""							
							curve_difference = p_recovery.linspace()[1][idx_r_start:idx_r_end] - p_baseline.linspace()[1][idx_b_start:idx_b_end]
							curves_sum = p_recovery.linspace()[1][idx_r_start:idx_r_end] + p_baseline.linspace()[1][idx_b_start:idx_b_end]
							histogram_difference = count_recovery[:, 1] - count_baseline[:, 1]
							
							
							x_plot = p_recovery.linspace()[0][idx_r_start:idx_r_end]
							
							curve_difference = curve_difference/len(strainObsKeys)							
							data_plot = np.abs(curve_difference)
							
							idx_12 = np.where(np.round(p_recovery.linspace()[0][idx_r_start:idx_r_end], 1) == 12.)[0][0] + 1
							
							max_absolute_difference = data_plot[:idx_12].max()
							idx_point = np.where( data_plot==max_absolute_difference )[0]
							x_point = x_plot[idx_point]
							y_point = curve_difference[idx_point]
							
							dStore = np.array([self.lstate, x_point, w, nr, r])
							dataStore['strain%d' %self.strain] = np.vstack([dataStore['strain%d' %self.strain], dStore]) if dataStore['strain%d' %self.strain].size else dStore			
							
							if self.strain<maxStrainID:								
								x_mutant = x_point
								y_mutant = y_point
								
							else:								
								x_wt = x_point
								y_wt = y_point				
							
						else:
							continue		
						
					else:
						continue
				ax1.plot(y_wt, y_mutant, c=colors[ci], marker='.', markersize=110.0)#, alpha=0.8)
				data_x.append(y_wt)
				data_y.append(y_mutant)
				labels.append(str(self.lstate))
		
		ax1.set_ylabel('Recovery-Baseline $\mathregular{Zfhx3^{Sci{/}{+}}}$', fontweight='bold', fontsize=130, labelpad=30)
		
		
		xint = []
		locs, l = plt.xticks()
		for each in locs:
			xint.append(round(each,2))
		plt.xticks(xint)
		
		ax1.set_xticklabels(ax1.get_xticks(), fontweight='bold', fontsize=120)
		
		labels = [item.get_text() for item in ax1.get_xticklabels()]
		labels[0] = ''		
		
		ax1.set_xticklabels(labels, fontweight='bold', fontsize=120)
		
		ax1.xaxis.set_ticks_position('none')
		
		ax1.set_xlim([-0.025, 0.05])
		
		ax1.yaxis.set_ticks_position('none')
		
		yint = []
		locs, l = plt.yticks()
		for each in locs:
			yint.append(round(each,2))
		plt.yticks(yint)
		
		ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=120)
		
		labels = [item.get_text() for item in ax1.get_yticklabels()]
		labels[0] = ''		
		
		ax1.set_yticklabels(labels, fontweight='bold', fontsize=120)
		
		ax1.set_xlabel('Recovery-Baseline $\mathregular{Zfhx3^{{+}{/}{+}}}$', fontweight='bold', fontsize=130, labelpad=30)
		
		ax1.axhline(y=0., color='k', ls='--', lw=23.0, label='_nolegend_')
		ax1.axvline(x=0., color='k', ls='--', lw=23.0, label='_nolegend_')

		
		legend_properties = {'weight':'bold', 'size':130}			
		
		legend = plt.legend(['wakefulness', 'sleep'], bbox_to_anchor=(.4, 1.12), loc='upper left', borderaxespad=0., prop=legend_properties)
		frame = legend.get_frame().set_alpha(0)
					
		fname = 'strainAbsDifference.tiff'
		fname = os.path.join('./homostaticResponse/', fname)
		fig.savefig(fname, format='tiff', transparent=True, dpi=100)
		plt.close(fig)
		
		savemat('./homostaticResponse/data.mat', mdict={'data':dataStore, 'columnLabels':['lstate', 'xPeak', 'epochsWake', 'epochsNR', 'epochsR']})
    
    def computeDistribution(self, d):
		"""
		Function computing the discrete distribution (histogram) of each 
		latent state over the 24h in one hour bin
		"""
				
		"""
		Iterate through hours and compute histogram:
		"""
		count = np.zeros((len(np.arange(18)), 2), dtype=float)
		i = 0
		for t in np.arange(24):
			if t in self.recoveryHours:
				count[i, 0] = t
				count[i, 1] = len( np.where(d[:, d.shape[1]-1]==t)[0] )
				i += 1
				
		idx_recHours = []
		for i in self.recoveryHours:
			i = float(i)
			idx_recHours.append( np.where(count[:, 0] == i)[0][0] )
		count = count[idx_recHours, :]
							
		"""
		Fit curve on histogram : Polynomial fit
		"""
		p = Polynomial.fit(self.recoveryInd, count[:, 1], self.polydeg)
		pConv = p.convert(domain=[-1, 1])

				
		return count, p
		


if __name__ == "__main__":
	
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-refDir', help='Experiment path')
	parser.add_argument('-epoch', help='Epoch analysed')
	parser.add_argument('-deg', help='Degrees of polynomial')
	args = parser.parse_args()
	
	print 'Initialization..'
	model = homostaticResponseAbsDiff(args.refDir, args.epoch, args.deg)

	print 'Loading data..'
	model.loadData()

	print 'Computing circadian profiles..'
	model.combinedGroupHistograms()
