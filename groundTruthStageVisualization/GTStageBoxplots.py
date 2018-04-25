""" 
	THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
						FILE AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************
		

Script written for visualizing the per strain boxplots of the manually 
scored sleep stages.


*********************************  OUTPUT    *********************************

Output : a folder named as "boxplotsGT" including the per ground truth
		 sleep stage boxplot of the diferent mouse groups, as defined
		 by the manual scoring.
		 		
*****************************************************************************

<vkatsageorgiou@vassia-PC>
"""

import logging
import sys
import os

import numpy as np
from pylab import *

from ConfigParser import *

from numpy.random import RandomState
from scipy.io import loadmat, savemat
from scipy import signal

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec


def boxplotStagePowerLstate(d, obsKeys, bands, labels, saveFigPath, lstate):
	'''
	Function for visualizing the boxplots of the initial EEG/EMG 
	data mapping to a latent state.
	'''
	
	plt.style.use('bmh')
	colors = list(plt.rcParams['axes.prop_cycle'])
	
	print(obsKeys)
			
	dEEG = [ d[:, j] for j in range( d.shape[1]-1 ) ]
	print(dEEG)
	dEMG = [ d[:, d.shape[1]-1] ]
	print(dEMG)
	
	fig = plt.figure(figsize=(13, 10))
	ax1 = fig.add_subplot(111)
	
	fig.suptitle('LS %d' %lstate, fontsize=40, fontweight='bold')
	
	bp = ax1.boxplot(dEEG, showfliers=False, patch_artist=True)
	
	ax1.grid(False)
	ax1.patch.set_facecolor('0.85')
	ax1.patch.set_alpha(0.5)
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_linewidth(5)
	ax1.spines['bottom'].set_color('k')
	ax1.spines['left'].set_linewidth(5)
	ax1.spines['left'].set_color('k')
	
	ax1.set_ylabel('Power', fontweight='bold', fontsize=30)
	
	ax1.xaxis.set_ticks_position('none')
	ax1.yaxis.set_ticks_position('none')
	
	ax1.set_ylim( [0, 3000] )
	
	yint = []
	locs, l = plt.yticks()
	for each in locs:
		yint.append(int(each))
	plt.yticks(yint)
	
	#ax1.set_ylim( [0, max(yint)] )	
	
	ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=25)
	
	xtickNames = ax1.set_xticklabels(bands[0:5])
	plt.setp(xtickNames, fontsize=30, fontweight='bold')
	
	
	plt.setp(bp['boxes'],            # customise box appearance
			 edgecolor='k',#colors[1],         # outline colour
			 linewidth=5.,             # outline line width
			 facecolor='None')     # fill box with colour
	plt.setp(bp['whiskers'], color='k', linestyle='--', linewidth=4.5)
	
	plt.setp(bp['medians'],          # customize median lines
			 color='k',#'#1a1a1a',            # line colour
			 linewidth=5.)             # line thickness
			 
			
	fname = 'LS%d' %lstate
	fname = os.path.join(saveFigPath, fname)
	fig.savefig(fname, transparent=True, dpi=100)
	plt.close(fig)


def boxplotStagePower(d, obsKeys, bands, labels, saveFigPath):
	'''
	Function for visualizing the boxplots of the LOG initial EEG/EMG 
	data mapping to eacch latent state.
	'''
	
	plt.style.use('bmh')
	colors = list(plt.rcParams['axes.prop_cycle'])
	
	print(obsKeys)
	
	for i in range(3):
		
		idx = np.where( obsKeys[:, 3] == i+1 )[0]
		print(idx)
		db = d[idx, :]
		
		dEEG = [ d[idx, j] for j in range( d.shape[1]-1 ) ]
		print(dEEG)
		dEMG = [ d[idx, d.shape[1]-1] ]
		print(dEMG)
		
		fig = plt.figure(figsize=(13, 10))
		ax1 = fig.add_subplot(111)
		
		fig.suptitle(labels[i], fontsize=40, fontweight='bold')
		
		bp = ax1.boxplot(dEEG, showfliers=False, patch_artist=True)
		
		ax1.grid(False)
		ax1.patch.set_facecolor('0.85')
		ax1.patch.set_alpha(0.5)
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_linewidth(5)
		ax1.spines['bottom'].set_color('k')
		ax1.spines['left'].set_linewidth(5)
		ax1.spines['left'].set_color('k')
		
		ax1.set_ylabel('Power', fontweight='bold', fontsize=30)
		
		ax1.xaxis.set_ticks_position('none')
		ax1.yaxis.set_ticks_position('none')
		
		ax1.set_ylim( [0, 3000] )
		
		yint = []
		locs, l = plt.yticks()
		for each in locs:
			yint.append(int(each))
		plt.yticks(yint)
		
		#ax1.set_ylim( [0, max(yint)] )
		
		ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=25)
		
		xtickNames = ax1.set_xticklabels(bands[0:5])
		plt.setp(xtickNames, fontsize=30, fontweight='bold')
		
		
		plt.setp(bp['boxes'],            # customise box appearance
				 edgecolor='k',#colors[1],         # outline colour
				 linewidth=5.,             # outline line width
				 facecolor='None')     # fill box with colour
		plt.setp(bp['whiskers'], color='k', linestyle='--', linewidth=4.5)
		
		plt.setp(bp['medians'],          # customize median lines
				 color='k',#'#1a1a1a',            # line colour
				 linewidth=5.)             # line thickness
				 
				
		fname = labels[i]
		fname = os.path.join(saveFigPath, fname)
		fig.savefig(fname, transparent=True, dpi=200)
		plt.close(fig)



def scaling(d):
	dMinRow = np.min(d, axis = 0)
	dMaxRow = np.max(d, axis = 0)
	
	d = 10.*((d - dMinRow) / (dMaxRow - dMinRow) - 0.5)
	
	return d
	
	
def main(scalingFlag):
	np.random.seed(124)
	prng =  RandomState(123)
	
	config = ConfigParser()
	config.read('DataVisual_config')
	data_path = config.get('Paths','data_path')
	file_name = config.get('Paths','file_name')
	
	dpp = DataPreproc()
	
	os.chdir(data_path)
	# Get current path:
	os.getcwd()
		
	# Import visible data :
	visData = file_name
	
	dataFile = np.load('../../' + visData)
	d = dataFile['bandsD']
	obsKeys = np.load('obsKeys.npz')['obsKeys'].astype(int)
	
	print d.shape
	
	#d = np.log(d + np.spacing(1))
	
	Bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'EMG']	
	labels = ['Wakefulness', 'NREM', 'REM']
	
	if not os.path.isdir('boxplotsGT'):
		os.makedirs('boxplotsGT')
		
	'''
	Scaling each feature seperatelly for visualization.
	'''
	if scalingFlag:
		print("Scaling..")
		d = scaling(d)
	
	# all dataset
	#boxplotStagePower(d, obsKeys, Bands, labels, './boxplotsGT/')
	
	"""
	Per strain
	"""
	
	"""
	Insert a column to label frames as strain : (1,2,3)
	"""
	obsKeys = np.insert(obsKeys, obsKeys.shape[1], 0, axis=1)
	for i in range(obsKeys.shape[0]):
		obsKeys[i, obsKeys.shape[1]-1] = int(str(obsKeys[i, obsKeys.shape[1]-2])[0])
		
	print( np.unique( obsKeys[:, obsKeys.shape[1]-1] ) )
	
	ind = np.where( obsKeys[:, obsKeys.shape[1]-1] == 2 )[0]
	print(ind.shape)
	print( obsKeys )
	
	
	# per strain
	boxplotStagePower(d[ind, :], obsKeys[ind, :], Bands, labels, './boxplotsGT/')
	
	"""
	Per lstate
	"""
	# ls = 91
	# ind = np.where(obsKeys[:, 1] == ls)[0]
	# print(len(ind))
	
	# # per lstate
	# boxplotStagePowerLstate(d[ind, :], obsKeys[ind, :], Bands, labels, './boxplotsGT/', ls)

	
if __name__ == "__main__":
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-scaling', help='scaling flag', default=False)
	args = parser.parse_args()
	
	main(args.scaling)	
