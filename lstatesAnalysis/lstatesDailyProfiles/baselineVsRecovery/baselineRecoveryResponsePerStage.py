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

here we test the homostatic response of the observed latent states in the
recovery mode (i.e. after sleep deprivation), comparing it to the baseline.


*********************************  OUTPUT    *********************************

Output : a folder named as "homostaticResponse" including the per latent
         state homostatic response profiles of the diferent mouse groups.
                 
*****************************************************************************
                 

<vkatsageorgiou@vassia-PC>
"""

from __future__ import division

import sys
import os
#from pylab import *
import numpy as np
import math
from numpy.random import RandomState
from numpy.polynomial import Polynomial
from scipy.io import loadmat, savemat
from configparser import *
import datetime
import PIL.Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib import use
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import MaxNLocator

from scipy.special import expit
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import stats
import _pickle as cPickle
import shutil


class homostaticResponse(object):    
    '''
    Object for analyzing the daily profiles of the observed latent states,
    testing the response of the latent states in the Recovery mode
    (i.e. after sleep deprivation).
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
        savemat('obsKeysTime.mat', mdict={'obsKeys':self.obsKeys})
    
    def combinedGroupHistograms(self):
        '''
        Method for visualizing the distribution of each latent state in time
        per group (strain).
        '''
        if not os.path.isdir('homostaticResponse'):
            os.makedirs('homostaticResponse')
                    
        
        self.baselineHours = [7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6]
        self.recoveryHours = [13, 13, 13, 13, 13, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6]
        self.baselineInd = np.arange(len(self.baselineHours))
        self.recoveryInd = np.arange(len(self.recoveryHours))
        
        plt.style.use('bmh')
        palete = plt.rcParams['axes.color_cycle']
        colors = ['#b2182b', palete[9], palete[8], '#1a1a1a', '#004529']
        
        width = 0.4
        
        
        """ Iterate through latent states """
        for self.lstate in self.lstates:
                
            idx = np.where(self.obsKeys[:,1]==self.lstate)[0]
            
            if len(idx) >= 100:
                
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(60,35))
                                        
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                
                ax1.spines['bottom'].set_linewidth(20)
                ax1.spines['bottom'].set_color('k')
                ax1.spines['left'].set_linewidth(20)
                ax1.spines['left'].set_color('k')
                
                ax1.yaxis.set_ticks_position('left')
                ax1.xaxis.set_ticks_position('bottom')
                ax1.grid(False)
                
                
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                
                ax2.spines['bottom'].set_linewidth(20)
                ax2.spines['bottom'].set_color('k')
                ax2.spines['left'].set_linewidth(20)
                ax2.spines['left'].set_color('k')
                ax2.yaxis.set_ticks_position('left')
                ax2.xaxis.set_ticks_position('bottom')
                ax2.grid(False)
                
                
                currObsKeys = self.obsKeys[idx,:]
                epochsW = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==1))[0]) / len(currObsKeys)), 3)            
                epochsNR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==2))[0]) / len(currObsKeys)), 3)
                epochsR = round((len(np.where((currObsKeys[:, currObsKeys.shape[1]-5]==3))[0]) / len(currObsKeys)), 3)                
                
                
                plt.text(0.01, 0.98, 'LS: ' + str(self.lstate), transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#252525', fontsize=150, fontweight='bold')
                                
                behaviorDistribution = [epochsW, epochsNR, epochsR]
                max_idx = behaviorDistribution.index(max(behaviorDistribution))
                
                if max_idx == 0:
                    plt.text(0.5, 1., 'W: ' + str(epochsW*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#b2182b', fontsize=150, fontweight='bold')
                    plt.text(0.5, 0.93, 'NR: ' + str(epochsNR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)
                    plt.text(0.5, 0.85, 'R: ' + str(epochsR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)                        
                elif max_idx == 1:
                    plt.text(0.5, 1., 'W: ' + str(epochsW*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)
                    plt.text(0.5, 0.93, 'NR: ' + str(epochsNR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#b2182b', fontsize=150, fontweight='bold')
                    plt.text(0.5, 0.85, 'R: ' + str(epochsR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)
                else:
                    plt.text(0.5, 1., 'W: ' + str(epochsW*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)
                    plt.text(0.5, 0.93, 'NR: ' + str(epochsNR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#1a1a1a', fontsize=150)
                    plt.text(0.5, 0.85, 'R: ' + str(epochsR*100) + '%', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#b2182b', fontsize=150, fontweight='bold')
                
                
                """ Iterate through mouse strains """                
                ci = 0
                maxStrainID = max(self.strainIDs)
                max_graph = 0.
                
                for self.strain in self.strainIDs:
                    idxStr = np.where(currObsKeys[:, currObsKeys.shape[1]-3]==self.strain)[0]
                    
                    if len(idxStr) != 0:
                        strainObsKeys = currObsKeys[idxStr, :]
                        
                        # spit into modes
                        baseline = strainObsKeys[np.where( strainObsKeys[:, strainObsKeys.shape[1]-2] == 1 )[0], :]
                        recovery = strainObsKeys[np.where( strainObsKeys[:, strainObsKeys.shape[1]-2] == 2 )[0], :]
                        
                        if (len(baseline) > 0. and len(recovery) > 0.):
                                    
                            count_baseline, p_baseline = self.computeDistribution(baseline, 'baseline')
                            count_recovery, p_recovery = self.computeDistribution(recovery, 'recovery')
                            
                            idx_b_start = np.where( p_baseline.linspace()[0] >= 6. )[0][0]
                            idx_b_end = np.where( p_baseline.linspace()[0] >= len(model.baselineHours)-6 )[0][0]
                            
                            idx_r_start = np.where( p_recovery.linspace()[0] >= 6. )[0][0]
                            idx_r_end = np.where( p_recovery.linspace()[0] >= len(model.recoveryHours)-6 )[0][0]
                                                        
                            if max(max(p_recovery.linspace()[1][idx_r_start:idx_r_end]), max(p_baseline.linspace()[1][idx_b_start:idx_b_end])) > max_graph:
                                max_graph = max(max(p_recovery.linspace()[1][idx_r_start:idx_r_end]), max(p_baseline.linspace()[1][idx_b_start:idx_b_end]))
                            
                            """ Compute curve difference """                            
                            
                            if self.strain<maxStrainID:
                                
                                l = '$\mathregular{Zfhx3^{Sci{/}{+}}}$'
                                
                                ax2.plot(*p_baseline.linspace(), c=colors[ci+1], linewidth=20.0, label='baseline')
                                ax2.plot(p_recovery.linspace()[0][idx_r_start:idx_r_end] + 6., p_recovery.linspace()[1][idx_r_start:idx_r_end], c=colors[ci], linewidth=20.0, label='recovery')
                                
                                t2 = ax2.set_title(l, fontsize=150, fontweight='bold', loc='left')
                                t2.set_position([0.0, 1.05])
                                ax2.axvspan(0., p_recovery.linspace()[0][idx_r_start]+6, facecolor='#b2182b', alpha=0.3)
                                ax2.text(1.21, .08, 'total: ' + str(len(strainObsKeys)) + ' epochs', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#252525', fontsize=150)
                            else:
                                l = '$\mathregular{Zfhx3^{{+}{/}{+}}}$'
                                
                                ax1.plot(*p_baseline.linspace(), c=colors[ci+1], linewidth=20.0, label='baseline')
                                ax1.plot(p_recovery.linspace()[0][idx_r_start:idx_r_end] + 6., p_recovery.linspace()[1][idx_r_start:idx_r_end], c=colors[ci], linewidth=20.0, label='recovery')
                                
                                t1 = ax1.set_title(l, fontsize=150, fontweight='bold', loc='left')        
                                t1.set_position([0.0, 1.05])
                                ax1.axvspan(0., p_recovery.linspace()[0][idx_r_start]+6, facecolor='#b2182b', alpha=0.3)    
                                ax1.text(0.01, .08, 'total: ' + str(len(strainObsKeys)) + ' epochs', transform=ax1.transAxes, horizontalalignment='left', verticalalignment='top', color='#252525', fontsize=150)    
                            
                        else:
                            continue        
                        
                    else:
                        continue                
                
                ax1.set_ylabel('Number of epochs', fontweight='bold', fontsize=150, labelpad=30)
                
                xTickMarks = []
                for j in np.arange(-6, 31):
                    if j in [0, 6, 12]:
                        xTickMarks.append('%s' %str(j))
                    elif j == 23:
                        xTickMarks.append('%s' %str(24))
                    else:
                        xTickMarks.append('')
                
                ax1.set_xticks(self.baselineInd+width/2)
                xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold', fontsize=130)
                
                ax1.xaxis.set_ticks_position('none')
                ax1.yaxis.set_ticks_position('none')
                
                ax1.set_ylim([0, int(math.ceil(max_graph / 10.0)) * 10])
                                
                yint = []
                locs, labels = plt.yticks()
                for each in locs:
                    yint.append(int(each))
                plt.yticks(yint)        
                
                ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=130)                
                                
                ax1.set_xlim([6, len(self.baselineHours)-6])
                ax1.set_xlabel('ZT', fontweight='bold', fontsize=150, labelpad=40)
                
                ax2.xaxis.set_ticks_position('none')
                ax2.yaxis.set_ticks_position('none')
                
                
                ax2.set_xticks(self.baselineInd+width/2)
                xtickNames = ax2.set_xticklabels(xTickMarks, fontweight='bold', fontsize=130)
                
                ax2.set_xlim([6, len(self.baselineHours)-6])
                ax2.set_xlabel('ZT', fontweight='bold', fontsize=150, labelpad=40)
                
                legend_properties = {'weight':'bold', 'size':130}            
                legend = plt.legend(bbox_to_anchor=(.5, 1.), loc='upper left', borderaxespad=0., prop=legend_properties)
                frame = legend.get_frame().set_alpha(0)
                
                                            
                fname = 'lstate%d.tiff' %self.lstate
                fname = os.path.join('./homostaticResponse/', fname)
                fig.savefig(fname, format='tiff', transparent=True, dpi=100)
                #plt.show()
                plt.close(fig)
    
    def computeDistribution(self, d, mode):
        """
        Function computing the discrete distribution (histogram) of each 
        latent state over the 24h in one hour bin
        """
        
        if mode == 'baseline' :
            
            """    Iterate through hours and compute histogram    """
            
            count = np.zeros((len(np.arange(24)), 2), dtype=float)
            for t in np.arange(24):
                count[t, 0] = t
                count[t, 1] = len( np.where(d[:, d.shape[1]-1]==t)[0] )
            
                        
            idx_baseHours = []
            for i in self.baselineHours:
                i = float(i)
                idx_baseHours.append( np.where(count[:, 0] == i)[0][0] )
            count = count[idx_baseHours, :]
                    
            """    Fit curve on histogram : Polynomial fit    """
            
            p = Polynomial.fit(self.baselineInd, count[self.baselineHours, 1], self.polydeg)
            pConv = p.convert(domain=[-1, 1])
            
        else:
            """    Iterate through hours and compute histogram """
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
                                
            """    Fit curve on histogram : Polynomial fit    """
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
    
    print ('Initialization..')
    model = homostaticResponse(args.refDir, args.epoch, args.deg)

    print ('Loading data..')
    model.loadData()

    print ('Computing circadian profiles..')
    model.combinedGroupHistograms()
    
    with open(args.refDir + 'doneHomoResp', 'w') as doneFile:
        doneFile.write(datetime.datetime.strftime(datetime.datetime.now(), '%d/%m/%Y %H:%M:%S'))
