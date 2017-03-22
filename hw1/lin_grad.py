# Machine Learning 2017
# Homework 1: 	Linear Regression
# Instructor: 	Hung-yi Lee
# Student: 		Michael Chiou
# Student ID: 	R05921086
# Email:		r05921086.ntu.edu.tw 
# Github: 		https://github.com/jastion/ML2017.git

import sys
import numpy as np
import matplotlib.pyplot as mpl
import time

#adjust print options to terminal
np.set_printoptions(linewidth=1e3, edgeitems=1e10)

class LineGradDesc:
	#This class contains functions to perform Gradient
	def __init__(self, inputTraining, inputTest, \
		inputFeatures,inputHours,inputVerify,):
		#Initializes Linear Gradient Descent Class
		#Preprocesses and sorts necessary data
		
		#save input data
		self.idxFeatures = inputFeatures #Which features to use
		self.rangeHours = inputHours #range of hours (0-9)
		self.percentVerify = inputVerify #% of data to store as Validation
		self.setTraining = inputTraining #Training Data set
		self.setTesting = inputTest #Testing Set


		mu = -1
		sigma = 0.5
		#Initialize Number of Features,  Weights, Bias
		self.numWeights = len(self.idxFeatures) * len(self.rangeHours) 
		self.weights = sigma* np.random.rand(self.numWeights,1) + mu 
		self.bias = sigma * np.random.rand(1,1) + mu

		#Preprocess and Sort Training and Testing Data 
		self.setTraining = self.sort_training_data()
		self.setTesting = self.sort_testing_data()
		
		#Normalize Training Data
		self.setTraining, meanI, stDevI= self.norm_features(self.setTraining)
		self.meanTraining = meanI
		self.stDevTraining = stDevI

		#Shuffle data
		np.random.shuffle(self.setTraining)
		
		#Get total number of available data points
		numData = self.setTraining.shape[0]
		#Get Index to segment data
		idxSegment = numData - int(numData*inputVerify)
		#print (idxSegment)

		#Segments total training data into validation and training data sets
		self.setValidation = self.setTraining[idxSegment:,:]
		self.setTraining = self.setTraining[:idxSegment,:]
		
		print ("Initialize Complete!")

	def sort_training_data(self):
		#Initializes variables to store sorted training data 
		#PM2.5 truth values will be appended as the last row to setTraining
		#This is an array which will be the shape of (5652, 163).
		setX = np.array([]).reshape(0,len(self.idxFeatures)*len(self.rangeHours))
		pm25 = np.array([]).reshape(0,(24*20-9)*12)

		#Iterates through data into N x Y where N is available 
		#data points and Y is number of features 
		for months in range(12):
		    for hours in range(self.rangeHours[0],24*20-len(self.rangeHours)):
		        temp = self.setTraining[self.idxFeatures[:,None],hours+months\
		        *480:hours+months*480+len(self.rangeHours)].\
		        flatten().reshape(1,len(self.idxFeatures)*len(self.rangeHours))
		        setX = np.vstack((setX,temp))#(5652,162)

		#Append the PM25 truth value to train_x_set
		for months in range(12):
			pm25 = np.append(pm25, self.setTraining[len(self.rangeHours),\
				len(self.rangeHours)+months*480:480+months*480])
		pm25 = pm25.reshape(setX.shape[0],1)#(5652,1)

		setTraining = np.append(setX,pm25,axis = 1)#(5652,163)
		return setTraining

	def sort_testing_data(self):
		#Sorts testing data into an array of size N x Y where N is available 
		#data points and Y is number of features  
		setTest = self.setTesting[self.idxFeatures[:,None],self.rangeHours]\
		.flatten().reshape(1,len(self.idxFeatures)*len(self.rangeHours))
		for days in range(1,12*20):
			setTest = np.vstack((setTest,self.setTesting[\
				self.idxFeatures[:,None],self.rangeHours+days*9].flatten()))

		return setTest


	def cost_fcn(self):
		#loss function for linear regression takes form of the following:
		#L(f) = sum (y_n - y) where y is a linear line
		
		setTrain = self.setTraining[:,:-1]
		setValidation = self.setValidation[:,:-1]
		
		yTrain= setTrain.dot(self.weights) + self.bias
		yValid = setValidation.dot(self.weights) + self.bias

		lossTrain = self.rmse(self.setTraining[:,-1], yTrain)
		lossValid = self.rmse(self.setValidation[:,-1], yValid)

		return lossTrain, lossValid
		
	def rmse(self, actual, predicted):
		#calculates root mean square for error tracking of form
		#sqrt( (sum of errors)^2)
		sumError = 0.0
		for i in range(len(actual)):
			predictionError = predicted[i] - actual[i]
			sumError += (predictionError ** 2)
		
		error = (sumError / float(len(actual))) 
		error = error ** 0.5
		'''
		if flagReg == 1:
			print("regularization")
			weightI = np.sum(self.weights) ** 0.5
			errorReg  = coeff * weightI
			errorTotal = errorReg + error 
			return errorTotal

		else:
			return error
		'''
		return error
	def norm_features(self, setInput):
		'''
		Returns a normalized version of X. Normalized X is calculated as follows
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

	def grad_desc(self, iterations, eta):
		self.eta = eta; #learning rate
		self.iterations = iterations #number of iterations

		#Initialize variables
		dwTotal = 1 #total weight differential
		dbTotal = 1 #total bias differential
		arrayValidError = np.array([]) #Validation error
		arrayTrainingError = np.array([])#Training error


		np.random.seed(1)
		#Get training data without truth values (last row)
		X = self.setTraining[:,:-1].reshape(self.setTraining.shape[0],\
										self.setTraining.shape[1]-1)#(4521,162)
		#Get truth values for trianing data
		yActual = self.setTraining[:,-1].reshape(self.setTraining.shape[0],1)#(4521,1)
		

		#lambdas = [0.5]
		#Iterate through gradient descent
		#for coeff in lambdas:
		#print (coeff)
		for idx in range(1,iterations+1):
			# Equation: y = b + sum(weights * X)
			dw = 0 #weight derivative
			db = 0 #bias derivative

			#calculate prediction
			yPredict = X.dot(self.weights) + self.bias
			#Determine error
			deltaError = yPredict - yActual

			#calculate error and bias errors
			tmpWeight = (2 * deltaError *(-X)).T
			tmpBias = 2 * deltaError * (-1)

			#determine derivatives
			diffWeight = np.sum(tmpWeight,1).reshape(tmpWeight.shape[0],1)
			diffBias = np.sum(tmpBias,0).reshape(1,1)

			#sum up total errors
			dwTotal += diffWeight**2
			dbTotal += diffBias**2

			coeffLamba = 0.5
			
			#update weights
			self.weights += (eta * diffWeight)/np.sqrt(dwTotal) #- (coeff * self.weights)
			self.bias += (eta * diffBias)/np.sqrt(dbTotal)
			
			#print training and validation losses
			if idx%1000 == 0 or idx == 1:
				errorValid, errorTrain = self.cost_fcn()
				print ("Iterations: %d Valid cost: %f Train cost: %f" % (idx,errorValid,errorTrain))
			
		print("Descent Complete!")
		return arrayValidError, arrayTrainingError

	def run_test_set(self):
		#Runs test set on trained weights
		setTesting = ((self.setTesting-self.meanTraining)/self.stDevTraining)
		prediction = setTesting.dot(self.weights)+self.bias

		#initialize variables to save to CSV files
		csvOutput = np.zeros((240+1,1+1), dtype ="|S6")
		#append labels and headers
		csvOutput[0,0] = "id"
		csvOutput[0,1] = "value"

		#append data
		for idx in range (240):
			csvOutput[idx+1,0] = "id_" + str(idx)
			csvOutput[idx+1,1] = float(prediction[idx,0])
		#Write data to CSV
		np.savetxt("./data/w_pm25.csv", self.weights, delimiter = ",", fmt = "%s")
		np.savetxt("./data/b_pm25.csv", self.bias, delimiter = ",", fmt = "%s")
		np.savetxt("./data/test_output.csv", csvOutput, delimiter=",", fmt = "%s")

		print("Save Complete!")

	def sigmoid(self,x):
		#Equation: 1/(1+e^-t)
		output = 1.0/(1.0+(np.exp(-x/c)))
		return output

	def sigmoid_output_to_derivative(self,output):
		#Sigmoid derivative function
		return output*(1-output)

	def neural_network(self, iterations, eta):
		#Non-implemented Neural Network function
		#Developer Notes:
		#	To be Implemented
		#		Back Propagation
		#		Error checking
		#		Different activation Functions
		#		Variable Hidden Layers?

		self.eta = eta;
		self.iterations = iterations
		#self.errorValid = np.array([]).reshape(iterations,1)
		#self.errorTrain = np.array([]).reshape(iterations,1)
		
		#Seed
		np.random.seed(1)
		#Collect Training and Ground Truth set
		X = self.setTraining[:,:-1].reshape(self.setTraining.shape[0],self.setTraining.shape[1]-1)#(4521,162)
		y = self.setTraining[:,-1].reshape(self.setTraining.shape[0],1)#(4521,1)

		#Collect Validation and Ground Truth set
		xValid = self.setValidation[:,:-1].reshape(self.setValidation.shape[0],self.setValidation.shape[1]-1)#(4521,162)
		yValid = self.setValidation[:,-1].reshape(self.setValidation.shape[0],1)#(4521,1)

		#Hidden Layer nodes
		hiddenLayer = 100
		#initialize synapses
		syn0 = 2*np.random.random((self.numWeights,hiddenLayer)) - 1
		#syn0 = 2*np.random.random((self.numWeights,1)) - 1
		syn1 = 2*np.random.random((hiddenLayer,1)) - 1	

		#initialize previous weights
		prevSynWeightUpdate0 = np.zeros_like(syn0)
		prevSynWeightUpdate1 = np.zeros_like(syn1)

		syn0DirCount = np.zeros_like(syn0)
		syn1DirCount = np.zeros_like(syn1)
		#alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]
		alphas = [0.0001]
		for alpha in alphas:
			#print (alpha)
			#Iterate over neural network
			for idx in xrange(iterations):
				layer0 = X
				layer0V = xValid

				layer1 = self.sigmoid(np.dot(layer0,syn0))
				layer2 = self.sigmoid(np.dot(layer1,syn1))

				layer1V = self.sigmoid(np.dot(layer0V,syn0))
				layer2V = self.sigmoid(np.dot(layer1V,syn1))

				errorLayer2V = yValid - layer2V
				errorLayer2 = y - layer2

				deltaLayer2 = errorLayer2*self.sigmoid_output_to_derivative(layer2)
				errorLayer1 = deltaLayer2.dot(syn1.T)
				deltaLayer1 = errorLayer1*self.sigmoid_output_to_derivative(layer1)

				#errorLayer1 = y - layer1
				if (idx % 1000) == 0:
					a = str(np.mean(np.abs(errorLayer2)))
					b = str(np.mean(np.abs(errorLayer1)))
					c = str(np.mean(np.abs(errorLayer2V)))
					print ("Error1:" + b + "   Error2: " + a + "    Validation: " + c)

				synWeightUpdate0 = layer1.T.dot(deltaLayer2)
				synWeightUpdate1 = layer2.T.dot(deltaLayer1)

				#if idx > 0:
			#		syn0DirCount += np.abs(((synWeightUpdate0 > 0)+0) - ((prevSynWeightUpdate0 > 0) + 0))
		#			syn1DirCount += np.abs(((synWeightUpdate1 > 0)+0) - ((prevSynWeightUpdate1 > 0) + 0))
				
				syn1 -= eta * synWeightUpdate1.T
				syn0 -= alpha * synWeightUpdate0.T

				prevSynWeightUpdate0 = synWeightUpdate0
				prevSynWeightUpdate1 = synWeightUpdate1

			#np.savetxt("./data/NN.csv", layer2, delimiter = ",", fmt = "%s")
		self.syn0 = syn0
		self.syn1 = syn1
		return 0

	def test_nn(self):
		#Test function to implement Neural network
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

		np.savetxt("./data/syn0_pm25.csv", syn0, delimiter = ",", fmt = "%s")
		np.savetxt("./data/syn1_pm25.csv", syn1, delimiter = ",", fmt = "%s")
		np.savetxt("./data/NN_output.csv", layer2, delimiter=",", fmt = "%s")
		return 0
	

