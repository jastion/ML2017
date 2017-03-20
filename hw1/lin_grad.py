import sys
import numpy as np
import matplotlib as mpl
import time
import csv
from sklearn.metrics import mean_squared_error

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

		#save input data
		self.idxFeatures = inputFeatures
		self.rangeHours = inputHours
		self.percentVerify = inputVerify
		self.setTraining = inputTraining
		self.setTesting = inputTest

		#Initialize extra variables
		self.numWeights = len(self.idxFeatures) * len(self.rangeHours)
		self.weights = np.random.rand(self.numWeights,1)
		self.bias = np.random.rand(1,1)

		#Preprocess Data
		self.setTraining, self.setTrainingPM25 = self.sort_training_data()
		self.setTesting, self.setTestingPM25 = self.sort_testing_data()
		self.setTraining, self.meanTraining, self.stDevTraining = self.norm_features(self.setTraining)

		#np.savetxt("output2.csv", self.setTraining,fmt="%s", delimiter=","), 
		
		np.random.shuffle(self.setTraining)
		#features*hours*
		#np.savetxt("output1.csv", self.setTraining,fmt="%s", delimiter=","),
		#print (self.setTraining.shape)
		
		#self.setTesting, self.meanTesting, self.stDevTesting = self.norm_features(self.setTesting)
		#print(self.setTraining.shape)
		numData = self.setTraining.shape[0]
		idxSegment = numData - int(numData*inputVerify)
		#print (idxSegment)

		self.setValidation = self.setTraining[idxSegment:,:]
		self.setTraining = self.setTraining[:idxSegment,:]

		#print (self.setTrainingPM25)
		self.setValidationPM25 = self.setTrainingPM25[idxSegment:,:]
		self.setTrainingPM25 = self.setTrainingPM25[:idxSegment,:]
		
		#print (self.setTrainingPM25.shape)
		#print (self.setValidationPM25.shape)
		#print (self.setTraining.shape)
		#print (self.setValidation.shape)
		print ("Initialize Complete!")

	def sort_training_data(self):
		csvData = self.setTraining
		#set NR 'No Rainfall' to 0
		#Copy data and convert NR to 0
		idxNans = np.isnan(csvData)
		csvData[idxNans] = 0
		
		csvData = csvData.clip(min=0)
		#Reorganize into an 18xN array
		csvData = np.vsplit(csvData,csvData.shape[0]/18)
		csvData = np.concatenate(csvData,1)
		#print(csvData.shape)
		#Pick selected features
		csvData = csvData[self.idxFeatures,:]
		csvData = csvData.T

		idxStart = 0;		
		idxEnd = 9;
		csvFinal = np.array([]).reshape(0,self.numWeights)
		pm25 = csvFinal
		while (idxEnd < csvData.shape[0]-1):
			csvTmp = csvData[idxStart:idxEnd,:].reshape(1,self.numWeights)
			csvFinal = np.vstack((csvFinal,csvTmp))
			pm25 = np.append(pm25,csvData[idxEnd+1,9])
			if ((idxEnd+1)%480 == 0):
				#print(idxStart, idxEnd)

				idxStart = idxEnd + 1
				idxEnd = idxStart + 9
				#print(idxStart, idxEnd)

			else:
				idxStart +=1
				idxEnd +=1
		#pm25 = csvData[,9]
		#outputs 5652,162
		#print(csvFinal)
		print (csvFinal.shape)
		print(pm25.shape)
		pm25 = pm25.reshape(pm25.shape[0],1)
		print(pm25.shape)
		#np.savetxt("csvFinal.csv", csvFinal,fmt="%s", delimiter=","), 
		return csvFinal, pm25 #[5651, 153] [5631, 1]

	def sort_testing_data(self):
		csvData = self.setTesting
		idxNans = np.isnan(csvData)
		csvData[idxNans] = 0
		#Reorganize into an Nx18 array
		csvData = np.vsplit(csvData,csvData.shape[0]/18)
		csvData = np.concatenate(csvData,1)
		#Pick selected features
		csvData = csvData[self.idxFeatures,:]
		csvData = csvData.T

		idxStart = 0;		
		idxEnd = 9;
		csvFinal = np.array([]).reshape(0,self.numWeights)
		pm25 = csvFinal

		while (idxEnd < csvData.shape[0]-1):
			csvTmp = csvData[idxStart:idxEnd,:].reshape(1,self.numWeights)
			csvFinal = np.vstack((csvFinal,csvTmp))
			pm25 = np.append(pm25,csvData[idxEnd+1,9])
			if ((idxEnd+1)%480 == 0):
				#print(idxStart, idxEnd)

				idxStart = idxEnd + 1
				idxEnd = idxStart + 9
				#print(idxStart, idxEnd)

			else:
				idxStart +=1
				idxEnd +=1
		#pm25 = csvData[,9]
		#outputs 5652,162
		print (csvFinal.shape)
		print(pm25.shape)
		pm25 = pm25.reshape(pm25.shape[0],1)
		print(pm25.shape)
		return csvFinal, pm25

	def cost_fcn(self):
		'''
		loss function for linear regression
		takes form of the following:
		L(f) = sum (y_n - y) where y is a linear line
		'''


		setTrain = self.setTraining
		setTrainPM25 = self.setTrainingPM25

		setValidationPM25 = self.setValidationPM25
		setValidation = self.setValidation

		yTrain= setTrain.dot(self.weights) + self.bias
		yValid = setValidation.dot(self.weights) + self.bias

		lossTrain = self.rmse(setTrainPM25, yTrain)
		lossValid = self.rmse(setValidationPM25, yValid)
		return lossTrain, lossValid
		
	def rmse(self, actual, predicted):
		sumError = 0.0
		for i in range(len(actual)):
			predictionError = predicted[i] - actual[i]
			sumError += (predictionError ** 2)
		error = (sumError / float(len(actual))) ** 0.5
		return error

	def norm_features(self, setInput):
		'''
		Returns a normalized version of X.
		normalized X is calculated as follows
		X_Norm = Xi - mean_i) / std_i
		This sets mean to 0
		'''
		meanI = []
		stDevI = []
		setNorm = setInput

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
		X = self.setTraining #Nx162
		yActual = self.setTrainingPM25
		
		dwTotal = 1
		dbTotal = 1
		for idx in range(1,iterations+1):
			dw = 0
			db = 0

			X = self.setTraining
			yPredict = X.dot(self.weights) + self.bias
			#print(yPredict.shape)
			deltaError = yPredict - yActual
			#print(deltaError.shape)
			tmpWeight = (2 * deltaError *(-X)).T
			tmpBias = 2 * deltaError * (-1)

			diffWeight = np.sum(tmpWeight,1).reshape(tmpWeight.shape[0],1)
			diffBias = np.sum(tmpBias,0).reshape(1,1)

			dwTotal += diffWeight**2
			dbTotal += diffBias**2
			self.weights += (eta * diffWeight)/np.sqrt(dwTotal)
			self.bias += (eta * diffBias)/np.sqrt(dbTotal)
			'''
			if (diffWeight < 0):
				self.weights += (eta * diffWeight)/np.sqrt(dwTotal)
			else:
				self.weights -= (eta * diffWeight)/np.sqrt(dwTotal)

			if (diffBias < 0):
				self.bias += (eta * diffBias)/np.sqrt(dbTotal)
			else:
				self.bias -= (eta * diffBias)/np.sqrt(dbTotal)
			'''
			if idx%1000 == 0 or idx == 1:
				valid_loss_error, train_loss_value = self.cost_fcn()
				print ("Iterations: %d Valid cost: %f Train cost:  %f" %(idx,valid_loss_error,train_loss_value))

		print("Descent Complete!")
		return 0

	def run_test_set(self):
		setTesting = ((self.setTesting-self.meanTraining)/self.stDevTraining)
		prediction = setTesting.dot(self.weights)+self.bias

		csvOutput = np.zeros((240+1,1+1), dtype ="|S6")
		csvOutput[0,0] = "id"
		csvOutput[0,1] = "value"

		for idx in range (240):
			csvOutput[idx+1,0] = "id_" + str(idx)
			csvOutput[idx+1,1] = float(prediction[idx,0])

		np.savetxt("./data/w_pm25.csv", self.weights, delimiter = ",", fmt = "%s")
		np.savetxt("./data/b_pm25.csv", self.bias, delimiter = ",", fmt = "%s")
		np.savetxt("./data/test_output.csv", csvOutput, delimiter=",", fmt = "%s")

		print("Save Complete!")