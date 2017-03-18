import sys
import numpy as np
import matplotlib as pl
import time
import csv

np.set_printoptions(linewidth=1e3, edgeitems=1e2)


# 0  AMB_TEMP		9	PM2.5  
# 1  CH4			10	RAINFALL
# 2  CO 			11	RH (Rel. Humidity)
# 3  NMHC			12	SO2
# 4  NO 			13 	THC
# 5  NO2 			14 	WD_HR
# 6  NOx 			15	WIND_DIRECT
# 7  O3				16	WIND_SPEED
# 8  PM10			17 	WS_HR

class LineGradDesc:

	def __init__(self, inputTraining, inputTest, \
		inputFeatures,inputHours,inputVerify):

		self.arrayFeatures = inputFeatures
		self.rangeHours = inputHours
		self.percentVerify = inputVerify

		self.setTraining = inputTraining
		self.setTest = inputTest
		self.weights = np.random.rand(len(self.arrayFeatures)*len(self.rangeHours),1)
		self.bias = np.random.rand(1,1)

		self.setTraining = self.sort_training_data()
		self.setTraining, self.mean, self.stDev = self.norm_features()
		np.random.shuffle(self.setTraining)


		#self.setTraining, self.mean, self.stDev = self.norm_features()
		
		#np.random.shuffle(self.train_data_set)

		'''
		self.setTraining = self.setTraining[:int(self.setTraining.shape[0]*(1-self.percentVerify)), :]
		self.setValidation = self.setTraining[:int(self.setTraining.shape[0]*self.percentVerify), :]
		
		#create test data set
		self.test_data_set = self.create_test_data()
		'''

		#print (self.arrayFeatures)
		#print (self.rangeHours)
		#print (self.weights.shape)
		#print (self.bias)
		#print (self.setTraining)
		print ("Initialize Complete!")

	def sort_training_data(self):
		csvData = self.setTraining
		#set NR 'No Rainfall' to 0
		#Copy data and convert NR to 0
		idxNans = np.isnan(csvData)
		csvData[idxNans] = 0
		
		#Reorganize into an 18xN array
		csvData = np.vsplit(csvData,csvData.shape[0]/18)
		csvData = np.concatenate(csvData,1)
		#Pick selected features
		csvData = csvData[self.arrayFeatures,:]
		print (("Training Shape is: %d %d") % (csvData.shape[0], csvData.shape[1]))
		np.savetxt("output.csv", csvData.T,fmt="%s", delimiter=","), 
		return csvData

	def sort_testing_data(self):
		csvData = self.setTest
		idxNans = np.isnan(csvData)
		csvData[idxNans] = 0
		#Reorganize into an 18xN array
		csvData = np.vsplit(csvData,csvData.shape[0]/18)
		csvData = np.concatenate(csvData,1)
		return csvData

	def cost_fcn(self):
		'''
		loss function for linear regression
		takes form of the following:
		L(f) = sum (y_n - y) where y is a linear line
		'''
		setTrain = self.setTrain
		setValidation = self.setValidation


		return 0
		
	def rmse_metric(actual, predicted):
		sum_error = 0.0
		for i in range(len(actual)):
			prediction_error = predicted[i] - actual[i]
			sum_error += (prediction_error ** 2)
		mean_error = sum_error / float(len(actual))
		return sqrt(mean_error)

	def norm_features(self):
		'''
		Returns a normalized version of X.
		normalized X is calculated as follows
		X_Norm = Xi - mean_i) / std_i
		This sets mean to 0
		'''
		meanI = []
		stDevI = []
		setInput = self.setTraining
		setNorm = self.setTraining

		for idx in range(setInput.shape[1]):
			tmpMean = np.nanmean(setInput[:, idx])
			tmpStDev = np.nanstd(setInput[:, idx])
			meanI.append(tmpMean)
			stDevI.append(tmpStDev)
			setNorm[:,idx] = ((setInput[:,idx] - tmpMean)/ tmpStDev)
		
		return setNorm, meanI, stDevI

	def adagrad(eta,time):
		etaNew = eta
		return etaNew

	def regularization():
		return 0


