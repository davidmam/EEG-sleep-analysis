"""
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice

Description : Strain Classification based on counts over latent states
				using all the stanrdard classifiers from sklearn.


@author : kats.vassia@gmail.com

Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
<vkatsageorgiou@vassia-PC>
"""


import logging
import sys
from sys import getsizeof
import numpy as np
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from pylab import *
import PIL.Image
import matplotlib.pyplot as plt
from ConfigParser import *
import os
#import mlpy
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
#from sklearn.cross_validation import KFold, StratifiedKFold
#from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
#from sklearn import tree
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier


class strainClassifier(object):	
    '''
    Object containing all the functions needed for classifing the strains.
	'''
	
    def __init__(self, refDir, normFlag):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # file with configuration details for the launched experiment
        #self.expConfigFilename = expConfigFilename
        # loading details from configuration files
        #self.loadExpConfig()
        
        self.normFlag = int(normFlag)
        print self.normFlag
        
        np.random.seed(124)
        self.prng =  RandomState(123)
    
    def loadExpConfig(self):
        '''
        Method loading the configuration details for the dataset to be tested
        '''
        config = ConfigParser()
        config.read(self.refDir + '/' + self.expConfigFilename)
        
        #tList = config.get('EXP_DETAILS','tId')
        #self.tId = [k for k in tList.split(',')]
        
        #-- Experiment details:
        self.expsDir = config.get('EXP_DETAILS','expsDir')
        self.expID = config.get('EXP_DETAILS','expID')
        self.epoch = config.getint('EXP_DETAILS','epoch')
        
        self.saveDir = self.expsDir + self.expID + 'analysis/epoch%d/' %self.epoch
    
    def classification(self):
		"""
		Object for controlling the functions.
		"""
		
		"""
		Move in current experiment folder
		"""
		os.chdir(self.refDir)
		os.getcwd()
		
		"""
		Load subjects' distribution over latent states
		"""
		countArray = loadmat('./groupStatistics/subjectsDistribution.mat')['subjectsDistribution']
		countOverStates = countArray[:, :countArray.shape[1]-2]
				
		"""
		Remove mouse specific latent-states
		"""
		idx_remove = []
		for i in range(countArray.shape[1]-2):# iterate through latent states
			#print i
			idx_elements = np.where( countArray[:, i] != 0)[0]
			if len(idx_elements) == 1:
				idx_remove.append(i)
		self.countArray = np.delete(countArray, idx_remove, axis=1)
		
		"""
		Define subjects ids
		"""
		self.subjects = self.countArray[:, self.countArray.shape[1]-2]
		self.strains = np.unique(self.countArray[:, self.countArray.shape[1]-1])
		
		"""
		"""
		if not os.path.isdir('standardClassifiers'):
			os.makedirs('standardClassifiers')
		os.chdir('standardClassifiers')
		
		print("We are in : ", os.getcwd())
		
		if self.normFlag==1:
			self.caseNormalizedHistograms()
		else:
			self.caseCountsHistograms()
    
    def caseCountsHistograms(self):
		"""
		Case 1: histograms based on counts over the latent states
		"""
		print("Case 1: histograms based on counts over the latent states..")
		class_output = np.zeros((len(self.subjects), 8), dtype=np.int32)
		counter = 0
		
		
		C_rbf, g_rbf, C_linear = self.gridSearchSVM(self.countArray[:, :self.countArray.shape[1]-2], self.countArray[:, self.countArray.shape[1]-1])
		
		with open ('Output.txt','a') as f:
			f.write("\n Kernel SVM parameters : C = %s, g = %s" %(C_rbf, g_rbf))
			f.write("\n Linear SVM parameters : C = %s" %C_linear)
			f.write("\n LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, Ensemble on counts.. ")
			
		for s in self.subjects:
			#self.subjectID = s
			dTest = self.countArray[self.countArray[:, self.countArray.shape[1]-2] == s, :]
			dTest = dTest[:, :dTest.shape[1]-2]
			
			dTrain = self.countArray[self.countArray[:, self.countArray.shape[1]-2] != s, :]
			yTrain = dTrain[:, dTrain.shape[1]-1]
			dTrain = dTrain[:, :dTrain.shape[1]-2]
			
			[LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout] = self.classifiers(dTrain, yTrain, dTest, C_rbf, g_rbf, C_linear)
			
			yOut = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
		
			#print("Subject %d classified as %s" %(s, yOut))
		
			with open ('Output.txt','a') as f:
				f.write("\n Subject %s classified as: %s, %s, %s, %s, %s, %s, %s" %(str(s), str(LDAout), str(kNNout), str(NBout), str(DTREEout), str(lSVMout), str(kSVMout), str(eclfout)))
			
			class_output[counter,0] = s
			class_output[counter,1:] = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
			
			counter += 1
		print class_output
		savemat('countClassif.mat', mdict={'countClassif':class_output})
    
    def caseNormalizedHistograms(self):
		
		"""
		Case 2: normalizing histograms
		"""
		print("Case 2: normalizing histograms..")
		class_output = np.zeros((len(self.subjects), 8), dtype=np.int32)
		counter = 0
				
		C_rbf, g_rbf, C_linear = self.gridSearchSVM(self.countArray[:, :self.countArray.shape[1]-2]/self.countArray[:, :self.countArray.shape[1]-2].sum(axis=1)[:,None], self.countArray[:, self.countArray.shape[1]-1])
		
		with open ('Output2.txt','a') as f:
			f.write("\n Kernel SVM parameters : C = %s, g = %s" %(C_rbf, g_rbf))
			f.write("\n Linear SVM parameters : C = %s" %C_linear)
			f.write("\n LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, Ensemble on histograms.. ")
			
		for s in self.subjects:
			#self.subjectID = s
			dTest = self.countArray[self.countArray[:, self.countArray.shape[1]-2] == s, :]
			dTest = dTest[:, :dTest.shape[1]-2]
			
			'''
			Compute subjects probability distribution
			'''
			dTest = dTest.astype(np.float32) + 0.005
			#dTest = dTest/dTest.sum()
			dTest = dTest/np.linalg.norm(dTest, 2)
			
			dTrain = self.countArray[self.countArray[:, self.countArray.shape[1]-2] != s, :]
			#yTrain = dTrain[:, dTrain.shape[1]-1]
			
			dTrain = dTrain.astype(np.float32)
			
			'''
			'''
			dTrainFinal = np.array([])
			yTrain = []
			for group in self.strains.astype(np.float32):
				idx_group = np.where( dTrain[:, dTrain.shape[1]-1] == group )[0]
				#dTrain[idx_group, :dTrain.shape[1]-2] = dTrain[idx_group, :dTrain.shape[1]-2]/dTrain[idx_group, :dTrain.shape[1]-2].sum()
				
				dCurrent = dTrain[idx_group, :dTrain.shape[1]-2]
				dCurrent = dCurrent.sum(axis=0) + 0.005
				
				#dCurrent = dCurrent/dCurrent.sum()
				dCurrent = dCurrent/np.linalg.norm(dCurrent, 2)	
				
				dTrainFinal = np.vstack([dTrainFinal, dCurrent]) if dTrainFinal.size else dCurrent
				yTrain.append(group.astype(np.int32))
			yTrain = np.asarray(yTrain)
			
			print dTrainFinal.sum(axis=1), dTrainFinal.shape
			print yTrain
			print dTest.sum(axis=1), dTest.shape
			
			[LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout] = self.classifiers(dTrainFinal, yTrain, dTest, C_rbf, g_rbf, C_linear)
			
			yOut = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
		
			#print("Subject %d classified as %s" %(s, yOut))
		
			with open ('Output2.txt','a') as f:
				f.write("\n Subject %s classified as: %s, %s, %s, %s, %s, %s, %s" %(str(s), str(LDAout), str(kNNout), str(NBout), str(DTREEout), str(lSVMout), str(kSVMout), str(eclfout)))
			
			class_output[counter,0] = s
			class_output[counter,1:] = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
			
			counter += 1
		print class_output
		savemat('histogramClassif.mat', mdict={'histogramClassif':class_output})
    
    def classifiers(self, dTrain, yTrain, dTest, C_rbf, g_rbf, C_linear):
		"""
		Method performing the classification using the 6 standard classifiers.
		"""
		
		""" LDA Classifier """
		#LDA(n_components=None, priors=None)
		clf = LDA(n_components=None, priors=None)
		clf.fit(dTrain, yTrain)		
		LDAout = clf.predict(dTest)
		
		""" kNN Classifier """
		#KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, **kwargs)
		# weights=' distance', 'uniform'
		neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto')
		neigh.fit(dTrain, yTrain)
		kNNout = neigh.predict(dTest)
		
		""" Naive-Bayes Classifier """
		nb = GaussianNB()
		nb.fit(dTrain, yTrain)
		NBout = nb.predict(dTest)
		#print("Naive-Bayes Class Prior : ", nb.class_prior_)
		#print("Naive-Bayes mean of each feature per class : ", nb.theta_)
		#print("Naive-Bayes variance of each feature per class : ", nb.sigma_)
		
		""" DecisionTree Classifier """
		# DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, ..
		# random_state=None, min_density=None, compute_importances=None, max_leaf_nodes=None)
		dtree = DecisionTreeClassifier(criterion='gini',random_state=self.prng)
		dtree.fit(dTrain, yTrain)
		DTREEout = dtree.predict(dTest)
		#print("DecisionTree Features Importance : ", dtree.feature_importances_)
		
		""" Linear SVM """
		# LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
		lsvm = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=C_linear, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
		lsvm.fit(dTrain, yTrain)
		lSVMout = lsvm.predict(dTest)
		#print("LinearSVM Weights : ", lsvm.coef_)
		
		""" kernel SVM """
		# LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
		ksvm = svm.SVC(C=C_rbf, kernel='rbf', degree=3, gamma=g_rbf, coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
		ksvm.fit(dTrain, yTrain)
		kSVMout = ksvm.predict(dTest)
		#print("KernelSVM Coefficients : ", ksvm.dual_coef_)
		
		#eclf = VotingClassifier(estimators=[('kNN', neigh), ('lda', clf), ('lsvm', lsvm)], voting='hard')# for cd1, c57, mixed
		eclf = VotingClassifier(estimators=[('kNN', neigh), ('lsvm', lsvm), ('ksvm', ksvm)], voting='hard')#,  weights=[2,2,1])
		#eclf = VotingClassifier(estimators=[('dtree', dtree), ('ksvm', ksvm), ('lsvm', lsvm)], voting='hard', weights=[2,3,1])
		eclf = eclf.fit(dTrain, yTrain)
		eclfout = eclf.predict(dTest)
		
		return [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
    
    def gridSearchSVM(self, X, y):
		"""
		"""
		C_range = np.logspace(-2, 10, 13)
		gamma_range = np.logspace(-9, 3, 13)
		
		""" rbf SVM """
		param_grid = dict(gamma=gamma_range, C=C_range)
		
		grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=LeaveOneOut())
		
		grid.fit(X, y)
		
		print("The best parameters are %s with a score of %0.2f"
				% (grid.best_params_, grid.best_score_))
		
		C_rbf = grid.best_params_['C']
		g_rbf = grid.best_params_['gamma']
		
		""" Linear SVM """
		grid = GridSearchCV(svm.LinearSVC(), param_grid={'C': C_range}, cv=LeaveOneOut())
		
		grid.fit(X, y)
		
		print("The best parameters are %s with a score of %0.2f"
				% (grid.best_params_, grid.best_score_))
		
		C_linear = grid.best_params_['C']
		
		return C_rbf, g_rbf, C_linear
		


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('-refDir', help='Analysis directory')
	parser.add_argument('-normFlag', help='Use probabilities instead of counts: 1 for yes, 0 for no.')
	args = parser.parse_args()
	
	#expFile = 'exp_details'

	print 'Initialization..'
	model = strainClassifier(args.refDir, args.normFlag)

	print 'Classification..'
	model.classification()
