""" 
	THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
						FILE AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************
		

Script written for detecting the peaks of per latent state profiles and 
visualizing the profiles of the latent states that have peaks in the light 
phase (REM-like analysis, see Fig 5 in the paper).

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
from matplotlib.collections import LineCollection
from matplotlib import use

from scipy.special import expit
from scipy.optimize import curve_fit

from scipy import stats
import cPickle

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

import sys; sys.path.insert(0, './lib/')
from peakdetect import peakdetect



class peaksDetector(object):	
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
		#savemat('obsKeysTime.mat', mdict={'obsKeys':self.obsKeys})
    
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
    
    def detectCurvePeaks(self):
		'''
		Method for visualizing the distribution of each latent state in time
		including all the animals together.
		'''
		
		if not os.path.isdir('dailyProfiles'):
			os.makedirs('dailyProfiles')
		os.chdir('dailyProfiles')
		
		if not os.path.isdir('curvesPeaks'):
			os.makedirs('curvesPeaks')
		
		timeList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		
		ind = np.arange(len(timeList))
		indnew = np.arange(0, len(timeList)-1, 0.1)
		
		plt.style.use('bmh')
		colors = ['#b2182b', '#238b45', '#3690c0', '#023858']	
		
		figAll = plt.figure(figsize=(27,20), frameon=False)
		ax = figAll.add_subplot(111)
		
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['bottom'].set_color('k')
		ax.spines['left'].set_linewidth(10)
		ax.spines['left'].set_color('k')
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('bottom')
		ax.grid(False)
		
		marker_style = dict(linestyle=':', color="#b2182b", markersize=10)
		
		width = 0.6
		
		"""
		Iterate through latent states:
		"""
		overAllarray = np.zeros( (len(timeList), len(self.lstates)), dtype=int )
		curvePoints = np.zeros( (len(self.lstates), len(timeList)), dtype=np.float32 )
		numPeaks = np.zeros( (len(self.lstates), 2), dtype=int )
		peaksArray = np.array([])
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
			
			
			""" Fit polynomial curve on histogram """
			p = Polynomial.fit(ind, count[timeList,1], self.polydeg)
			pConv = p.convert(domain=[-1, 1])
			
			""" Find points on curve """
			yCurve = self.findCurvePoints(ind, count[timeList,1], pConv.coef)
			curvePoints[lstate, :] = yCurve
			
			y = yCurve[6:len(timeList)-6]
			
			max_peaks, min_peaks = peakdetect(y, lookahead=1)
			if len(currObsKeys) > 89:
				mylist = [epochsW, epochsNR, epochsR]
				if mylist.index(max(mylist)) != 0:# if lstate NREM or REM							
					if len(max_peaks):
						for mpeak in max_peaks:
							if mpeak[0] <= 12:
								xs = np.array([lstate, mpeak[1], mpeak[0], len(currObsKeys), len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==2))[0]), \
									len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==3))[0]), len(np.where((currObsKeys[:, currObsKeys.shape[1]-4]==1))[0]), 
									epochsNR, epochsR, epochsW])
								peaksArray = np.vstack([peaksArray, xs]) if peaksArray.size else xs
								
								if mpeak[0] < 5:
									ax.plot(*p.linspace(), c='#4575b4', linewidth=10.0)
									
								if (mpeak[0] >= 8 and mpeak[0] < 11):
									ax.plot(*p.linspace(), c='#081d58', linewidth=10.0)
							
			
			num_picks = len(max_peaks) + len(min_peaks)
			
			# if lstate has at least 500 epochs:
			if len(idx) >= 500:
				with open ('./curvesPeaks/peaks.txt', 'a') as f:
					f.write('\nlstate %d : %d ' %(lstate, num_picks))
			
			numPeaks[lstate, 0] = lstate
			numPeaks[lstate, 1] = num_picks	
			
			
		#savemat('./curvesPeaks/lstateTimeCount.mat', mdict={'lstateTimeCount':overAllarray})
		#savemat('./curvesPeaks/curvePoints.mat', mdict={'curvePoints':curvePoints})
		savemat('./curvesPeaks/numPeaks.mat', mdict={'numPeaks':numPeaks})
		
		peaksDict = {}
		peaksDict['data'] = peaksArray
		peaksDict['columnLabel'] = ['lstate', 'peak', 'idxPeak', 'numEpochs', 'numEpochsNREM', 'numEpochsREM', 'numEpochsWAKE', 'percentageNREM', 'percentageREM', 'percentageWAKE']
		savemat('./curvesPeaks/peaks.mat', mdict={'peaks':peaksDict})
		
		ax.set_ylabel('Frequency as %', fontweight='bold', fontsize=68)
		ax.set_xlabel('ZT', fontweight='bold', fontsize=68)
		ax.xaxis.set_ticks_position('none')
		ax.yaxis.set_ticks_position('none')
		
		yint = []
		locs, labels = plt.yticks()
		for each in locs:
			yint.append(round(each, 2))
		plt.yticks(yint)
			
		ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=68)
		
		xTickMarks = []
		for j in np.arange(-6, 31):
			if j in [0, 6, 12]:
				xTickMarks.append('%s' %str(j))
			elif j == 23:
				xTickMarks.append('%s' %str(24))
			else:
				xTickMarks.append('')
		ax.set_xticks(ind+width/2)
		xtickNames = ax.set_xticklabels(xTickMarks, fontweight='bold', fontsize=68)
		
		ax.set_xlim([6, len(timeList)-17])
		ax.set_ylim([0.0, 0.09])
		
					
		fname = 'curvesPeaksLightPhase.tiff'
		fname = os.path.join('./curvesPeaks/', fname)
		figAll.savefig(fname, format='tiff', transparent=True, dpi=100)
		plt.close(figAll)
    
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
	model = peaksDetector(args.f, args.epoch, args.deg, args.case)

	print 'Loading data..'
	model.loadData()

	print 'Detecting curve peaks..'
	model.detectCurvePeaks()
