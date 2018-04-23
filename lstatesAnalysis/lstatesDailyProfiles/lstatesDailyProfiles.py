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
(from a trained RBM model) latent states for the study of sleep in mice.


*********************************  OUTPUT    *********************************

Output : a folder named as "dailyProfiles" including the per latent
		 state daily profiles of the diferent mouse groups.
		 		
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
from matplotlib import use


class dailyProfiles(object):	
    '''
    Object for analyzing the daily profiles of
    the observed latent states.
	'''
	
    def __init__(self, refDir, epochID, polydeg, case):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # id of the epoch to be analysed
        self.epochID = int(epochID)
        # degree of polynomial
        self.polydeg = int(polydeg)
        
        self.case = case
                
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
		self.epochTime = dataFile['epochTime']
		
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
		Insert a column to label frames as strain : (1,2,3)
		"""
		self.obsKeys = np.insert(self.obsKeys, self.obsKeys.shape[1], 0, axis=1)
		for i in range(self.obsKeys.shape[0]):
			self.obsKeys[i, self.obsKeys.shape[1]-1] = int(
								str(self.obsKeys[i, self.obsKeys.shape[1]-2])[0])			
		
		self.lstates = np.unique( self.obsKeys[:, 1] )
		self.subjIDs = np.unique( self.obsKeys[:, self.obsKeys.shape[1]-2] )
		#print self.subjIDs
		self.strainIDs = np.unique( self.obsKeys[:, self.obsKeys.shape[1]-1] )
		
		self.epochTime = self.epochTime[:self.obsKeys.shape[0], :]
		
		self.utc_transform()
		
		"""
		Create unique array: epochsIDs - lstatesIDs - subjects - dayTime
		"""
		self.obsKeys = np.hstack((self.obsKeys, self.epochTime[:, self.epochTime.shape[1]-1].reshape(-1,1))).astype(int)
		
    
    def utc_transform(self):
		'''
		Method for transforming the serial date-time to integer.
		'''
		self.epochTime = np.insert(self.epochTime, self.epochTime.shape[1], 0, axis=1)
		
		"""
		Iterate through epochs:
		"""
		for i in range(len(self.epochTime)):
			self.epochTime[i, self.epochTime.shape[1]-1] = int(
				datetime.datetime.utcfromtimestamp( ( 
					self.epochTime[i, self.epochTime.shape[1]-2]-25569) * 86400.0).strftime("%H"))
    
    def createHistograms(self):
		'''
		Method for visualizing the distribution of each latent state in time
		including all the animals together.
		'''
		
		if not os.path.isdir('dailyProfiles'):
			os.makedirs('dailyProfiles')
		os.chdir('dailyProfiles')
		if not os.path.isdir('overAll'):
			os.makedirs('overAll')
		
		timeList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		
		ind = np.arange(len(timeList))
		indnew = np.arange(0, len(timeList)-1, 0.1)
		
		plt.style.use('bmh')
		colors = ['#b2182b', '#238b45', '#3690c0', '#023858']		
		
		"""
		Iterate through latent states:
		"""
		overAllarray = np.zeros( (len(timeList), len(self.lstates)), dtype=int )
		
		for lstate in self.lstates:			
			
			idx = np.where(self.obsKeys[:,1]==lstate)[0]
			currObsKeys = self.obsKeys[idx,:]
			
			epochsW = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==1))[0]) / len(currObsKeys)), 3)			
			epochsNR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==2))[0]) / len(currObsKeys)), 3)
			epochsR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==3))[0]) / len(currObsKeys)), 3)	
			
			"""
			Iterate through hours and compute histogram:
			"""
			count = np.zeros((len(timeList), 2), dtype=float)
			for t in timeList:
				count[t, 0] = t
				count[t, 1] = len( np.where(currObsKeys[:, currObsKeys.shape[1]-1]==t)[0] )
				overAllarray[t, lstate] = len( np.where(currObsKeys[:, currObsKeys.shape[1]-1]==t)[0] )
			
			count[:,1] = count[:,1]/np.sum(count[:,1])
			
			"""
			Plot
			"""		
			fig = plt.figure(figsize=(15,15), frameon=False)
			ax1 = fig.add_subplot(111)
			fig.suptitle('Latent-state %d \nTotal number of epochs: %d \nAwake: %s, NREM: %s, REM: %s' %(lstate, len(currObsKeys), str(epochsW*100), str(epochsNR*100), str(epochsR*100)), fontsize=20, fontweight='bold')
		
			ax1.spines['top'].set_visible(False)
			ax1.spines['right'].set_visible(False)
			ax1.spines['bottom'].set_visible(False)
			ax1.spines['left'].set_visible(False)
			ax1.yaxis.set_ticks_position('left')
			ax1.xaxis.set_ticks_position('bottom')
			width = 0.6
			
			ax1.bar(ind, count[timeList,1], width, color=colors[0], edgecolor = "none")
			#ax1.plot(x + width/2, y, c='g')
			
			"""
			Fit curve on histogram
			"""
			'''
			Polynomial fit:
			'''
			p = Polynomial.fit(ind, count[timeList,1], self.polydeg)
			pConv = p.convert(domain=[-1, 1])
			
			#err = self.computeError(ind, count[timeList,1], pConv.coef)
			#with open ('error.txt','a') as f:
			#	f.write('\nLatent-state %d : %s ' %(lstate, str(err*100)) + "%")
			
			ax1.plot(*p.linspace(), c='k', linewidth=3.0)
						
			ax1.set_ylabel('Frequency as %', fontweight='bold', fontsize=20)
			ax1.set_xlabel('Time', fontweight='bold', fontsize=20)
			xTickMarks = ['%s' %str(j) for j in timeList]
			ax1.set_xticks(ind+width/2)
			xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold')
			#plt.setp(xtickNames, rotation=90, fontsize=15)
			plt.setp(xtickNames, fontsize=15)
			ax1.xaxis.set_ticks_position('none')
			ax1.yaxis.set_ticks_position('none')		
			ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=17)
			ax1.set_xlim([6, len(timeList)-6])
						
			plt.close(fig)	 
			
		savemat('lstateTimeCount.mat', mdict={'lstateTimeCount':overAllarray})
	
    def combinedGroupHistograms(self):
		'''
		Method for visualizing the distribution of each latent state in time
		per group (strain).
		'''
		
		if not os.path.isdir('dailyProfiles'):
			os.makedirs('dailyProfiles')
		os.chdir('dailyProfiles')
		if not os.path.isdir('plots'):
			os.makedirs('plots')
		
		savemat('obsKeysTime.mat', mdict={'obsKeys':self.obsKeys})
		
		timeList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		ind = np.arange(len(timeList))
		indnew = np.arange(0, len(timeList)-1, 0.1)
		
		plt.style.use('bmh')
		palete = plt.rcParams['axes.color_cycle']
		colors = ['#b2182b', palete[9], palete[8], '#1a1a1a', '#004529']
		
		"""
		Iterate through latent states:
		"""
		for lstate in self.lstates:
				
			idx = np.where(self.obsKeys[:,1]==lstate)[0]
			currObsKeys = self.obsKeys[idx,:]
			epochsW = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==1))[0]) / len(currObsKeys)), 3)			
			epochsNR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==2))[0]) / len(currObsKeys)), 3)
			epochsR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==3))[0]) / len(currObsKeys)), 3)	
			
			fig = plt.figure(figsize=(33,30), frameon=False)
			ax1 = fig.add_subplot(111)
			
			fig.suptitle('LS ' + str(lstate) + ' - Total: ' + str(len(currObsKeys)) + ' epochs' + 
							'\nWAKE: ' + str(epochsW*100) + '%, NREM: ' + str(epochsNR*100) + '%, REM: ' + 
							str(epochsR*100) + '%', fontsize=80, fontweight='bold')
		
			ax1.spines['top'].set_visible(False)
			ax1.spines['right'].set_visible(False)
			ax1.spines['bottom'].set_linewidth(15)
			ax1.spines['bottom'].set_color('k')
			ax1.spines['left'].set_linewidth(15)
			ax1.spines['left'].set_color('k')
			ax1.yaxis.set_ticks_position('left')
			ax1.xaxis.set_ticks_position('bottom')
			ax1.grid(False)
			width = 0.6
			
			"""
			Iterate through strains:
			"""
			ci = -1
			maxStrainID = max(self.strainIDs)
			for strain in self.strainIDs:
				ci += 1
				idxStr = np.where(currObsKeys[:, currObsKeys.shape[1]-2]==strain)[0]
				if len(idxStr) != 0:
					strainObsKeys = currObsKeys[idxStr, :]			
					"""
					Iterate through hours and compute histogram:
					"""
					count = np.zeros((len(timeList), 2), dtype=float)
					for t in timeList:
						count[t, 0] = t
						count[t, 1] = len( np.where(strainObsKeys[:, strainObsKeys.shape[1]-1]==t)[0] )
					
					count[:,1] = count[:,1]/np.sum(count[:,1])
					
					"""
					Plot
					"""					
					ax1.scatter(ind, count[timeList,1], s=80, c=colors[ci], edgecolors = "none")
					
					"""
					Fit curve on histogram
					"""
					'''
					Polynomial fit:
					'''
					p = Polynomial.fit(ind, count[timeList,1], self.polydeg)
					pConv = p.convert(domain=[-1, 1])
			
					
					
					if strain<maxStrainID:
						if 'Zfhx3' in self.case:
							l = '$\mathregular{Zfhx3^{Sci{/}{+}}}$'
						else:
							if strain==1:
								l = 'Mixed'
							else:
								l = 'CD1'
					else:
						if 'Zfhx3' in self.case:
							l = '$\mathregular{Zfhx3^{{+}{/}{+}}}$'
						else:
							l = 'C57BL/6J'
					ax1.plot(*p.linspace(), c=colors[ci], linewidth=15.0, label=l)
					
				else:
					continue
				
			ax1.set_ylabel('Frequency as %', fontweight='bold', fontsize=80)
			ax1.set_xlabel('ZT', fontweight='bold', fontsize=80)
						
			xTickMarks = []
			for j in np.arange(-6, 31):
				if j in [0, 6, 12]:
					xTickMarks.append('%s' %str(j))
				elif j == 23:
					xTickMarks.append('%s' %str(24))
				else:
					xTickMarks.append('')
			
			ax1.set_xticks(ind+width/2)
			xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold', fontsize=80)
			
			ax1.xaxis.set_ticks_position('none')
			ax1.yaxis.set_ticks_position('none')
			
			yint = []
			locs, labels = plt.yticks()
			for each in locs:
				yint.append(round(each, 2))
			plt.yticks(yint)	
				
			ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=75)
			ax1.set_xlim([6, len(timeList)-6])
		
			legend_properties = {'weight':'bold', 'size':75}			
			legend = plt.legend(bbox_to_anchor=(.77, 1.), loc=2, borderaxespad=0., prop=legend_properties)
			frame = legend.get_frame().set_alpha(0)
						
			fname = 'lstate%d.png' %lstate
			fname = os.path.join('./plots/', fname)
			fig.savefig(fname, transparent=True, dpi=100)
			plt.close(fig)
    



if __name__ == "__main__":
	
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', help='Experiment path')
	parser.add_argument('-done', help='Experiment done flag', default=False)
	parser.add_argument('-epoch', help='Epoch analysed')
	parser.add_argument('-case', help='Strains analysed')
	parser.add_argument('-deg', help='Strains analysed')
	args = parser.parse_args()
	
	print 'Initialization..'
	model = dailyProfiles(args.f, args.epoch, args.deg, args.case)

	print 'Loading data..'
	model.loadData()

	print 'Computing daily profiles..'
	#model.createHistograms()
	model.combinedGroupHistograms()
	
	with open(args.f + 'doneDailyProfiles', 'w') as doneFile:
		doneFile.write(datetime.datetime.strftime(datetime.datetime.now(), '%d/%m/%Y %H:%M:%S'))
