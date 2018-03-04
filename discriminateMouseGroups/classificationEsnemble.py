"""
	THIS CODE IS UNDER THE MIT LICENSE. YOU CAN FIND THE COMPLETE FILE
					AT THE SOURCE DIRECTORY.
					
	Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
	
	@author : vasiliki.katsageorgiou@gmail.com
	
	
						Publication:
	A Novel Unsupervised Analysis of Electrophysiological
		Signals Reveals New Sleep Sub-stages in Mice
		
		
*****************************************************************************


						DESCRIPTION:				
This script can be used when performing multi-subject analysis
in order to test whether the different mouse genetic backgrounds
are easily distinguishable.

Each subject is associated with a discrete probability
distribution over the k=1,2,...,n observed latent states.

Classification is performed with a leave one subject out schema.
An ensemble of classifiers based on a linear Support Vector Machine (SVM), 
a Linear Discriminant Analysis (LDA) and a one-Nearest Neighbor (1-NN),
is used to classify the left out subject as belonging to one of the
under analysis mouse groups.


<vkatsageorgiou@vassia-PC>
"""


import logging
import os
import sys
from sys import getsizeof
import numpy as np
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier


class mouseGroupDiscrimination(object):	
    '''
    Object performing discrimination of the different mouse genotypes
    used in the experiment.
	'''
	
    def __init__(self, refDir, normFlag):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir     
        # flag whether to normalize the histograms   
        self.normFlag = int(normFlag)
        
        np.random.seed(124)
        self.prng =  RandomState(123)
    
    def classification(self):
				
		os.chdir(self.refDir)
		os.getcwd()
		
		""" Load subjects' distribution over the latent states """
		countArray = loadmat('./groupStatistics/subjectsDistribution.mat')['subjectsDistribution']
		countOverStates = countArray[:, :countArray.shape[1]-2]
				
		""" Remove mouse specific latent-states """
		idx_remove = []
		for i in range(countArray.shape[1]-2):# iterate through latent states
			idx_elements = np.where( countArray[:, i] != 0)[0]
			if len(idx_elements) == 1:
				idx_remove.append(i)
		self.countArray = np.delete(countArray, idx_remove, axis=1)
		
		""" Get subjects ids """
		self.subjects = self.countArray[:, self.countArray.shape[1]-2]
		self.strains = np.unique(self.countArray[:, self.countArray.shape[1]-1])
		
		if not os.path.isdir('strainsClassification'):
			os.makedirs('strainsClassification')
		os.chdir('strainsClassification')
		
		if self.normFlag==1:
			self.caseNormalizedHistograms()
		else:
			self.caseCountsHistograms()
    
    def caseCountsHistograms(self):
		""" Using histograms based on the counts over the latent states """
		
		print("Using histograms based on counts over the latent states..")
		class_output = np.zeros((len(self.subjects), 8), dtype=np.int32)
		counter = 0
		
		# perform grid search for SVM parameters
		C_rbf, g_rbf, C_linear = self.gridSearchSVM(self.countArray[:, :self.countArray.shape[1]-2], self.countArray[:, self.countArray.shape[1]-1])
		
		with open ('classification.txt','a') as f:
			f.write("\n Kernel SVM parameters : C = %s, g = %s" %(C_rbf, g_rbf))
			f.write("\n Linear SVM parameters : C = %s" %C_linear)
			f.write("\n LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, Ensemble on counts.. ")
			
		# iterating through subjects
		for s in self.subjects:
			# dTest = histogram of the test subject
			dTest = self.countArray[self.countArray[:, self.countArray.shape[1]-2] == s, :]
			dTest = dTest[:, :dTest.shape[1]-2]
			
			# dTrain = histograms of the rest of test subjects
			dTrain = self.countArray[self.countArray[:, self.countArray.shape[1]-2] != s, :]
			yTrain = dTrain[:, dTrain.shape[1]-1]
			dTrain = dTrain[:, :dTrain.shape[1]-2]
			
			[LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout] = self.classifiers(dTrain, yTrain, dTest, C_rbf, g_rbf, C_linear)
			
			yOut = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
		
			with open ('classification.txt','a') as f:
				f.write("\n Subject %s classified as: %s, %s, %s, %s, %s, %s, %s" %(str(s), str(LDAout), str(kNNout), str(NBout), str(DTREEout), str(lSVMout), str(kSVMout), str(eclfout)))
			
			class_output[counter,0] = s
			class_output[counter,1:] = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
			
			counter += 1
			
		print class_output
		savemat('countClassif.mat', mdict={'countClassif':class_output})
    
    def caseNormalizedHistograms(self):
		""" Using normalized histograms based on the counts over the latent states """
		
		print("Using normalized histograms based on the counts over the latent states..")
		class_output = np.zeros((len(self.subjects), 8), dtype=np.int32)
		counter = 0
		
		# perform grid search for SVM parameters				
		C_rbf, g_rbf, C_linear = self.gridSearchSVM(self.countArray[:, :self.countArray.shape[1]-2]/self.countArray[:, :self.countArray.shape[1]-2].sum(axis=1)[:,None], self.countArray[:, self.countArray.shape[1]-1])
		
		with open ('classification_normHists.txt','a') as f:
			f.write("\n Kernel SVM parameters : C = %s, g = %s" %(C_rbf, g_rbf))
			f.write("\n Linear SVM parameters : C = %s" %C_linear)
			f.write("\n LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, Ensemble on histograms.. ")
		
		# iterating through subjects	
		for s in self.subjects:
			# dTest = histogram of the test subject
			dTest = self.countArray[self.countArray[:, self.countArray.shape[1]-2] == s, :]
			dTest = dTest[:, :dTest.shape[1]-2]
			
			dTest = dTest.astype(np.float32) + 0.005
			dTest = dTest/np.linalg.norm(dTest, 2)
			
			# dTrain = histograms of the rest of test subjects			
			dTrain = self.countArray[self.countArray[:, self.countArray.shape[1]-2] != s, :]			
			dTrain = dTrain.astype(np.float32)
			
			dTrainFinal = np.array([])
			yTrain = []
			for group in self.strains.astype(np.float32):
				idx_group = np.where( dTrain[:, dTrain.shape[1]-1] == group )[0]
				
				dCurrent = dTrain[idx_group, :dTrain.shape[1]-2]
				dCurrent = dCurrent.sum(axis=0) + 0.005
				
				dCurrent = dCurrent/np.linalg.norm(dCurrent, 2)	
				
				dTrainFinal = np.vstack([dTrainFinal, dCurrent]) if dTrainFinal.size else dCurrent
				yTrain.append(group.astype(np.int32))
			yTrain = np.asarray(yTrain)
			
			
			[LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout] = self.classifiers(dTrainFinal, yTrain, dTest, C_rbf, g_rbf, C_linear)
			
			yOut = [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
		
			#print("Subject %d classified as %s" %(s, yOut))
		
			with open ('classification_normHists.txt','a') as f:
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
		clf = LDA(n_components=None, priors=None)
		clf.fit(dTrain, yTrain)		
		LDAout = clf.predict(dTest)
		
		""" kNN Classifier """
		neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto')
		neigh.fit(dTrain, yTrain)
		kNNout = neigh.predict(dTest)
		
		""" Naive-Bayes Classifier """
		nb = GaussianNB()
		nb.fit(dTrain, yTrain)
		NBout = nb.predict(dTest)
		
		""" DecisionTree Classifier """
		dtree = DecisionTreeClassifier(criterion='gini',random_state=self.prng)
		dtree.fit(dTrain, yTrain)
		DTREEout = dtree.predict(dTest)
		
		""" Linear SVM """
		lsvm = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=C_linear, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
		lsvm.fit(dTrain, yTrain)
		lSVMout = lsvm.predict(dTest)
		
		""" kernel SVM """
		ksvm = svm.SVC(C=C_rbf, kernel='rbf', degree=3, gamma=g_rbf, coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
		ksvm.fit(dTrain, yTrain)
		kSVMout = ksvm.predict(dTest)
		
		eclf = VotingClassifier(estimators=[('kNN', neigh), ('lsvm', lsvm), ('ksvm', ksvm)], voting='hard')#,  weights=[2,2,1])
		eclf = eclf.fit(dTrain, yTrain)
		eclfout = eclf.predict(dTest)
		
		return [LDAout, kNNout, NBout, DTREEout, lSVMout, kSVMout, eclfout]
    
    def gridSearchSVM(self, X, y):
		""" Method performing grid-search for SVM parameters """
		
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

	print 'Initialization..'
	model = mouseGroupDiscrimination(args.refDir, args.normFlag)

	print 'Classification..'
	model.classification()
