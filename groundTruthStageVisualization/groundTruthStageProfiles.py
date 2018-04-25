""" 
	THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
						FILE AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************
		

Script written for analysing the per strain daily profiles in the manually 
scored sleep stages.


*********************************  OUTPUT    *********************************

Output : a folder named as "dailyProfilesGT" including the per ground truth
		 sleep stage daily profiles of the diferent mouse groups, as defined
		 by the manual scoring.
		 		
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

import sys; sys.path.insert(0, './lib/')
from peakdetect import peakdetect


class dailyProfiles(object):	
    '''
    Object containing all the functions needed for the analysis of the
    day-time that each latent state occurs.
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
		
		if not os.path.isdir('dailyProfilesGT'):
			os.makedirs('dailyProfilesGT')
		os.chdir('dailyProfilesGT')
    
    def utc_transform(self):
		'''
		Method for transforming the serial date-time to integer.
		'''
		self.epochTime = np.insert(self.epochTime, self.epochTime.shape[1], 0, axis=1)
		"""
		Iterate through epochs:
		"""
		for i in range(len(self.epochTime)):
			self.epochTime[i, self.epochTime.shape[1]-1] = int(datetime.datetime.utcfromtimestamp((self.epochTime[i, self.epochTime.shape[1]-2]-25569) * 86400.0).strftime("%H"))
    
	
    def combinedGroupHistograms(self):
		'''
		Method for visualizing the distribution of each GT stage in time
		including all the animals together.
		'''
		
		
		"""
		Find classes
		"""
		self.stages = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-4])
		self.stageLabels = ['Wakefulness', 'NREM Sleep', 'REM Sleep']
					
		idxToStore = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
		timeList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		
		ind = np.arange(len(timeList))
		indnew = np.arange(0, len(timeList)-1, 0.1)
		
		plt.style.use('bmh')
		palete = plt.rcParams['axes.color_cycle']
		colors = ['#b2182b', palete[9], palete[8], '#1a1a1a', '#004529']
		
		"""
		Iterate through stages:
		"""
		curvePoints = {}
		numPeaks = {}
		timeHistogram = {}
		peaksArray = {}
		peaksArray['columnLabel'] = ['stage', 'peak', 'idxPeak', 'numEpochs']
		for strain in self.strainIDs:
			curvePoints['strain%d' %strain] = np.zeros( (len(idxToStore), len(self.stages)), dtype=np.float32 )
			numPeaks['strain%d' %strain] = np.zeros( (len(self.stages), 2), dtype=int )
			timeHistogram['strain%d' %strain] = np.zeros( (len(idxToStore), len(self.stages)), dtype=int )
			peaksArray['strain%d' %strain] = np.array([])
		
		"""
		Iterate through stages:
		"""
		for stage in self.stages:
				
			idx = np.where(self.obsKeys[:, self.obsKeys.shape[1]-4]==stage)[0]
			currObsKeys = self.obsKeys[idx, :]
			
			
			fig = plt.figure(figsize=(33,30), frameon=False)
			ax1 = fig.add_subplot(111)
			fig.suptitle(self.stageLabels[stage-1], fontsize=80, fontweight='bold')
			
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
					timeHistogram['strain%d' %strain][:, stage-1] = count[idxToStore, 1]
					
					count[:,1] = count[:,1]/np.sum(count[:,1])
					
					"""
					Plot
					"""					
					ax1.scatter(ind, count[timeList, 1], s=80, c=colors[ci], edgecolors = "none")
					
					"""
					Fit curve on histogram
					"""
					'''
					Polynomial fit:
					'''
					p = Polynomial.fit(ind, count[timeList,1], self.polydeg)
					pConv = p.convert(domain=[-1, 1])
					
					'''
					Find points on curve
					'''
					yCurve = self.findCurvePoints(ind, count[timeList,1], pConv.coef)
					curvePoints['strain%d' %strain][:, stage-1] = yCurve[6:len(timeList)-6]
					
					y = yCurve[6:len(timeList)-6]					
					
					max_peaks, min_peaks = peakdetect(y, lookahead=1)
					num_picks = len(max_peaks) + len(min_peaks)
								
					if len(max_peaks):
						for mpeak in max_peaks:
							xs = np.array([stage, mpeak[1], mpeak[0], len(strainObsKeys)])
							peaksArray['strain%d' %strain] = np.vstack([peaksArray['strain%d' %strain], xs]) if peaksArray['strain%d' %strain].size else xs
					
					
					numPeaks['strain%d' %strain][stage-1, 0] = stage
					numPeaks['strain%d' %strain][stage-1, 1] = num_picks	
					
					
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
					
					xm = [point[0] + 6 for point in max_peaks]
					ym = [point[1] for point in max_peaks]
					xn = [point[0] + 6 for point in min_peaks]
					yn = [point[1] for point in min_peaks]
										
					#self.plotBarPlots(self.stageLabels[stage-1], l, strain, count, p, xm, ym, xn, yn)
					
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
			xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold', fontsize=75)			
			
			plt.setp(xtickNames, fontsize=80)
			ax1.xaxis.set_ticks_position('none')
			ax1.yaxis.set_ticks_position('none')
			
			yint = []
			locs, labels = plt.yticks()
			for each in locs:
				yint.append(round(each, 2))
			plt.yticks(yint)
			ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=65)
			ax1.set_xlim([6, len(timeList)-6])
			
			legend_properties = {'weight':'bold', 'size':75}			
			legend = plt.legend(bbox_to_anchor=(.77, 1.), loc=2, borderaxespad=0., prop=legend_properties)
			frame = legend.get_frame().set_alpha(0)
						
			fname = 'stage%d.tiff' %stage
			fig.savefig(fname, format='tiff', transparent=True, dpi=100)
			plt.close(fig)
		savemat('numPeaks.mat', mdict={'numPeaks':numPeaks})
		
		for strain in self.strainIDs:
			timeHistogram['strain%d' %strain] = np.insert(timeHistogram['strain%d' %strain], 0, idxToStore, axis=1)
			curvePoints['strain%d' %strain] = np.insert(curvePoints['strain%d' %strain], 0, idxToStore, axis=1)
		np.savez('curvePoints.npz', curvePoints=curvePoints, numPeaks=numPeaks, timeHistogram=timeHistogram)
		
		savemat('timeHistogram.mat', mdict={'timeHistogram':timeHistogram})
		savemat('curvePoints.mat', mdict={'curvePoints':curvePoints})
		savemat('peaks.mat', mdict={'peaks':peaksArray})
    
    def plotBarPlots(self, stage, strain, strainID, count, p, xm, ym, xn, yn):
		"""
		Method for ploting the per strain histogram as bar plot
		"""
		
		timeList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		
		ind = np.arange(len(timeList))
		
		plt.style.use('bmh')
		colors = ['#b2182b', '#238b45', '#3690c0', '#023858']
		
		fig = plt.figure(figsize=(15,15), frameon=False)
		ax1 = fig.add_subplot(111)
		fig.suptitle(strain + ' - ' + stage, fontsize=20, fontweight='bold')
	
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.spines['left'].set_visible(False)
		ax1.yaxis.set_ticks_position('left')
		ax1.xaxis.set_ticks_position('bottom')
		width = 0.6
		
		ax1.bar(ind+width/2., count[timeList,1], width, color=colors[0], edgecolor = "none")
		
		ax1.plot(*p.linspace(), c='k', linewidth=3.0)
		
		for point in range(len(xm)):
			ax1.plot(xm[point], ym[point], marker='*', markersize=30, color="blue")
		for point in range(len(xn)):
			ax1.plot(xn[point], yn[point], marker='*', markersize=30, color="blue")			
		
		ax1.set_ylabel('Frequency as %', fontweight='bold', fontsize=20)
		ax1.set_xlabel('Time', fontweight='bold', fontsize=20)
		xTickMarks = ['%s' %str(j) for j in timeList]
		ax1.set_xticks(ind+width/2)
		xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold')
		
		plt.setp(xtickNames, fontsize=15)
		ax1.xaxis.set_ticks_position('none')
		ax1.yaxis.set_ticks_position('none')		
		ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=17)
		ax1.set_xlim([6, len(timeList)-6])
					
		fname = 'strain%d' %strainID + stage + '.png' 
		
		fig.savefig(fname, transparent=True, dpi=100)
		plt.close(fig)
	
    
    def findCurvePoints(self, x, y, c):
		"""
		Object for computing the points on the curve fitted to the bar
		graph.
		"""
		yCurve = []
		for xi in x:
			yi = self.polynomialFunct(c, xi)
			
			yCurve.append( yi )
		
		return np.asarray(yCurve)    
    
    def polynomialFunct(self, c, x):
		"""
		Polynomial function
		"""
		y = c[0]
		for i in range(1, len(c)):
			y += c[i]*(x**i)
		
		return y    
    


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

	print 'Computing GT stage profiles..'
	model.combinedGroupHistograms()
	
	with open(args.f + 'doneDailyProfilesGT', 'w') as doneFile:
		doneFile.write(datetime.datetime.strftime(datetime.datetime.now(), '%d/%m/%Y %H:%M:%S'))
