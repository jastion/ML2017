import sys
import numpy as np
import matplotlib as mpl
import time
import csv


np.set_printoptions(linewidth=1e3, edgeitems=1e10)


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
		#self.bias = np.array([20]).reshape(1,1)
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
			pm25 = np.append(pm25, self.setTraining[len(self.rangeHours),len(self.rangeHours)+months*480:480+months*480])
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
		#self.errorValid = np.array([]).reshape(iterations,1)
		#self.errorTrain = np.array([]).reshape(iterations,1)
		dwTotal = 1
		dbTotal = 1
		valid_loss_error = 0
		train_loss_error = 0

		np.random.seed(1)
		X = self.setTraining[:,:-1].reshape(self.setTraining.shape[0],self.setTraining.shape[1]-1)#(4521,162)
		yActual = self.setTraining[:,-1].reshape(self.setTraining.shape[0],1)#(4521,1)
		
		for idx in range(1,iterations+1):
			dw = 0
			db = 0


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
				errorValid, errorTrain = self.cost_fcn()
				print ("Iterations: %d Valid cost: %f Train cost:  %f" %(idx,errorValid,errorTrain))
				#self.errorValid = errorValid
				#self.errorTrain = errorTrain
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

	def sigmoid(self,x):

		output = 1.0/(1.0+(np.exp(-x)))
		return output

	def sigmoid_output_to_derivative(self,output):
		return output*(1-output)

	def neural_network(self, iterations, eta):
		self.eta = eta;
		self.iterations = iterations
		#self.errorValid = np.array([]).reshape(iterations,1)
		#self.errorTrain = np.array([]).reshape(iterations,1)
		dwTotal = 1
		dbTotal = 1
		valid_loss_error = 0
		train_loss_error = 0
		#np.linalg.norm(self.setTraining)
		np.random.seed(1)
		X = self.setTraining[:,:-1].reshape(self.setTraining.shape[0],self.setTraining.shape[1]-1)#(4521,162)
		X.astype("longfloat")
		#print(X.shape)
		y = self.setTraining[:,-1].reshape(self.setTraining.shape[0],1)#(4521,1)
		#print("X Shape: " + str(X.shape))
		#print(X[:2,:])

		xValidation = self.setValidation[:,:-1]

		np.random.seed(1)
		hiddenLayer = 5
		syn0 = 2*np.random.random((self.numWeights,int(np.mean(self.numWeights)))) - 1
		#syn0 = 2*np.random.random((self.numWeights,1)) - 1
		syn1 = 2*np.random.random((int(np.mean(self.numWeights)),1)) - 1	

		prevSynWeightUpdate0 = np.zeros_like(syn0)
		prevSynWeightUpdate1 = np.zeros_like(syn1)

		syn0DirCount = np.zeros_like(syn0)
		syn1DirCount = np.zeros_like(syn1)

		#alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]
		alphas = [0.0001]
		for alpha in alphas:
			#print (alpha)
			for idx in xrange(iterations):
				
				layer0 = X
				layer0V = 0
				#np.savetxt("NN.csv", np.dot(layer0,syn0), delimiter = ",", fmt = "%s")

				layer1 = self.sigmoid(np.dot(layer0,syn0))
				#print("layer1 shape:" + str(layer1.shape))
				#print(layer2.shape)
				#print(syn1.shape)
				layer2 = self.sigmoid(np.dot(layer1,syn1))

				errorLayer2 = y - layer2

				deltaLayer2 = errorLayer2*self.sigmoid_output_to_derivative(layer2)
				#print("delta layer2: " + str(deltaLayer2.shape))
				errorLayer1 = deltaLayer2.dot(syn1.T)
				#errorLayer1 = y - layer1
				if (idx % 1000) == 0:
					print ("Error:" + str(np.mean(np.abs(errorLayer1))))
				deltaLayer1 = errorLayer1*self.sigmoid_output_to_derivative(layer1)

				#print("Layer 1 Shape: " + str(layer1.shape))
				#print("layer 2 shape: " + str(layer2.shape))
				#print("syn0 shape:" + str(syn0.shape))
				#print("syn1 shape:" + str(syn1.shape))
				synWeightUpdate0 = layer1.T.dot(deltaLayer2)
				synWeightUpdate1 = layer2.T.dot(deltaLayer1)
				#print("syn update0: " + str(synWeightUpdate0.shape))
				#print("syn update1: " + str(synWeightUpdate1.shape))

				#if idx > 0:
			#		syn0DirCount += np.abs(((synWeightUpdate0 > 0)+0) - ((prevSynWeightUpdate0 > 0) + 0))
		#			syn1DirCount += np.abs(((synWeightUpdate1 > 0)+0) - ((prevSynWeightUpdate1 > 0) + 0))
				
				#print (idx)
				#print ("syn1 shape: " +str(syn1.shape))
				#print (synWeightUpdate1.shape)
				syn1 -= eta * synWeightUpdate1.T
				syn0 -= alpha * synWeightUpdate0.T

				prevSynWeightUpdate0 = synWeightUpdate0
				prevSynWeightUpdate1 = synWeightUpdate1
				'''
				layer0 = X
				layer1 = self.sigmoid(np.dot(layer0,syn0))

				errorLayer1 = layer1 - y

				sumError = 0.0
				for i in range(len(errorLayer1)):
					predictionError = errorLayer1[i]
					sumError += (predictionError ** 2)
				
				error = (sumError / float(len(errorLayer1))) 
				error = error ** 0.5
				if idx%10 == 0 or idx == 1:
					print("Error is: " + str(error))
				deltaLayer1 = errorLayer1 * self.sigmoid_output_to_derivative(layer1)
				syn0Deriv = np.dot(layer0.T,deltaLayer1)

				syn0 += alpha*syn0Deriv
				'''
			#np.savetxt("./data/NN.csv", layer2, delimiter = ",", fmt = "%s")
		self.syn0 = syn0
		self.syn1 = syn1
		return 0

	def test_nn(self):
		setTesting = ((self.setTesting-self.meanTraining)/self.stDevTraining)
		#prediction = setTesting.dot(self.weights)+self.bias

		syn0 = self.syn0
		syn1 = self.syn1

		layer0 = setTesting
		layer1 = self.sigmoid(np.dot(layer0,syn0))
		layer2 = self.sigmoid(np.dot(layer1,syn1))

		csvOutput = np.zeros((240+1,1+1), dtype ="|S6")
		csvOutput[0,0] = "id"
		csvOutput[0,1] = "value"

		for idx in range (240):
			csvOutput[idx+1,0] = "id_" + str(idx)
			csvOutput[idx+1,1] = float(layer2[idx,0])

		np.savetxt("./data/w_pm25.csv", self.weights, delimiter = ",", fmt = "%s")
		np.savetxt("./data/b_pm25.csv", self.bias, delimiter = ",", fmt = "%s")
		np.savetxt("./data/NN_output.csv", layer2, delimiter=",", fmt = "%s")
		return 0
		