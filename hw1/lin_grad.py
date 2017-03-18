import sys
import numpy as np
import matplotlib as mpl
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
		inputFeatures,inputHours,inputVerify,inputFolds, inputParameter):

		self.idxFeatures = inputFeatures
		self.rangeHours = inputHours
		self.percentVerify = inputVerify

		self.setTraining = inputTraining
		self.setTest = inputTest
		print (len(self.idxFeatures))
		print(len(self.rangeHours))
		self.numWeights = len(self.idxFeatures) * len(self.rangeHours)
		self.weights = np.random.rand(self.numWeights,1)
		print(self.weights.shape)
		self.bias = np.random.rand(1,1)

		self.setTraining = self.sort_training_data()
		self.setTest = self.sort_testing_data()

		
		#np.savetxt("output2.csv", self.setTraining,fmt="%s", delimiter=","), 
		np.random.shuffle(self.setTraining)
		#features*hours*
		#np.savetxt("output1.csv", self.setTraining,fmt="%s", delimiter=","),

		self.setTraining, self.mean, self.stDev = self.norm_features()

		self.setTraining = self.setTraining[int(self.setTraining.shape[0]*inputVerify):,:]
		self.setValidation = self.setTraining[:int(self.setTraining.shape[0] * inputVerify),:]

		self.setTrainingPM25 = self.setTraining[:,9]
		self.setValidationPM25 = self.setValidation[:,9]

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
		csvData = csvData[self.idxFeatures,:]
		csvData = csvData.T
		idxStart = 0;
		idxEnd = 9;
		csvFinal = np.array([]).reshape(0,self.numWeights)
		pm25 = csvFinal
		while (idxEnd < csvData.shape[0]):
			csvTmp = csvData[idxStart:idxEnd,:].reshape(1,162)
			csvFinal = np.vstack((csvFinal,csvTmp))
			pm25 = np.append(pm25,csvData[idxEnd+1,9])
			if (idxEnd%479 == 0):
				print(idxStart, idxEnd)

				idxStart = idxEnd + 1
				idxEnd = idxStart + 9
				print(idxStart, idxEnd)

			else:
				idxStart +=1
				idxEnd +=1
		#pm25 = csvData[,9]
		#outputs 5652,162
		print(csvFinal.shape)
		return csvFinal

	def sort_testing_data(self):
		csvData = self.setTest
		idxNans = np.isnan(csvData)
		csvData[idxNans] = 0
		#Reorganize into an Nx18 array
		csvData = np.vsplit(csvData,csvData.shape[0]/18)
		csvData = np.concatenate(csvData,1)
		return csvData.T

	def cost_fcn(self):
		'''
		loss function for linear regression
		takes form of the following:
		L(f) = sum (y_n - y) where y is a linear line
		'''
		setTrain = self.setTraining
		setValidation = self.setValidation
		setTrainPM25 = self.setTrainingPM25
		setValidationPM25 = self.setValidationPM25

		yValid = xValid.dot(self.weights) + self.bias
		yTrain= xTrain.dot(self.weights) + self.bias

		lossTrain = rmse(setTrainPM25, yTrain)
		lossValid = rmse(setValidationPM25, yValid)
		return lossTrain, lossValid
		
	def rmse(actual, predicted):
		sum_error = 0.0
		for i in range(len(actual)):
			predictionError = predicted[i] - actual[i]
			sumError += (predictionError ** 2)
		meanError = sumError / float(len(actual))
		return sqrt(meanError)

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

	def grad_desc(self, iterations, eta):
		self.eta = eta;
		self.iterations = iterations
		X = self.setTraining #Nx18
		idxStart = 0;
		idxEnd = 17;
		for idx in range(1,iterations+1):
			X = self.setTraining[idxStart:idxEnd,:].reshape(162,1)
			yPredict = X.dot(self.weights) + self.bias



			

		return 0

