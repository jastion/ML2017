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
		inputFeatures,inputHours,inputVerify,):

		#save input data
		self.idxFeatures = inputFeatures
		self.rangeHours = inputHours
		self.percentVerify = inputVerify
		self.setTraining = inputTraining
		self.setTesting = inputTest

		#Initialize extra variables
		self.numWeights = len(self.idxFeatures) * len(self.rangeHours)
		print("numWeights: %d" %(self.numWeights))
		self.weights = np.random.rand(self.numWeights,1)
		self.bias = np.random.rand(1,1)

		#Preprocess Data
		self.setTraining = self.sort_training_data()
		self.setTesting = self.sort_testing_data()
		
		self.setTraining, self.meanTraining, self.stDevTraining = self.norm_features(self.setTraining)
		#print (self.setTraining[0,:])
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

		#print (self.setTrainingPM25.shape)
		#print (self.setValidationPM25.shape)
		#print (self.setTraining.shape)
		#print (self.setValidation.shape)
		
		print ("Initialize Complete!")

	def sort_training_data(self):
		#read test_data file
		# This is an array which will be the shape of (5652, 163) and be returned.
		setX = np.array([]).reshape(0,len(self.idxFeatures)*len(self.rangeHours))
		pm25 = np.array([]).reshape(0,(24*20-9)*12)

		for months in range(12):
		    for hours in range(self.rangeHours[0],24*20-len(self.rangeHours)):
		        temp = self.setTraining[self.idxFeatures[:,None],hours+months*480:hours+months*480+len(self.rangeHours)].\
		        flatten().reshape(1,len(self.idxFeatures)*len(self.rangeHours))
		        setX = np.vstack((setX,temp))#(5652,162)

		#Append the correct pm25 value to train_x_set
		for months in range(12):
			pm25 = np.append(pm25, self.setTraining[9,9+months*480:480+months*480])
		pm25 = pm25.reshape(setX.shape[0],1)#(5652,1)
		setTraining = np.append(setX,pm25,axis = 1)#(5652,163)
		#np.savetxt("outputSetX.csv", setX,fmt="%s", delimiter=","), 
		return setTraining

	def sort_testing_data(self):
		setTest = self.setTesting[self.idxFeatures[:,None],self.rangeHours].flatten().reshape(1,len(self.idxFeatures)*len(self.rangeHours))
		for days in range(1,12*20):
			setTest = np.vstack((setTest,self.setTesting[self.idxFeatures[:,None],self.rangeHours+days*9].flatten()))

		return setTest



	def cost_fcn(self):
		'''
		loss function for linear regression
		takes form of the following:
		L(f) = sum (y_n - y) where y is a linear line
		'''
		setTrain = self.setTraining[:,:-1]

		setValidation = self.setValidation[:,:-1]
		
		yTrain= setTrain.dot(self.weights) + self.bias

		yValid = setValidation.dot(self.weights) + self.bias

		lossTrain = self.rmse(self.setTraining[:,-1], yTrain)
		lossValid = self.rmse(self.setValidation[:,-1], yValid)

		return lossTrain, lossValid
		
	def rmse(self, actual, predicted):
		sumError = 0.0
		for i in range(len(actual)):
			predictionError = predicted[i] - actual[i]
			sumError += (predictionError ** 2)
		
		error = (sumError / float(len(actual))) 
		error = error ** 0.5
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
		setNorm = setInput[:,:-1]

		for idx in range(setNorm.shape[1]):
			tmpMean = np.nanmean(self.setTraining[:, idx])
			tmpStDev = np.nanstd(self.setTraining[:, idx])
			meanI.append(tmpMean)
			stDevI.append(tmpStDev)
			setNorm[:,idx] = ((setNorm[:,idx] - tmpMean)/ tmpStDev)

		setNorm = np.append(setNorm,self.setTraining[:,-1].\
			reshape(self.setTraining.shape[0],1),1)

		return setNorm, meanI, stDevI

	def adagrad(eta,time):
		etaNew = eta
		return etaNew

	def regularization():
		return 0

	def grad_desc(self, iterations, eta):
		self.eta = eta;
		self.iterations = iterations
		
		dwTotal = 1
		dbTotal = 1
		valid_loss_error = 0
		train_loss_error = 0

		for idx in range(1,iterations+1):
			dw = 0
			db = 0
			X = self.setTraining[:,:-1].reshape(self.setTraining.shape[0],self.setTraining.shape[1]-1)#(4521,162)
			yActual = self.setTraining[:,-1].reshape(self.setTraining.shape[0],1)#(4521,1)

			yPredict = X.dot(self.weights) + self.bias

			deltaError = yPredict - yActual

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
		#print(self.meanTraining.shape)
		#print(self.stDevTraining.shape)
		#print(self.setTesting.shape)
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