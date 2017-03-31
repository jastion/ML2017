import numpy as np
import feature_classification as fc
import sys
import time
class LogDesc:
	#This class contains functions to perform Gradient
	def __init__(self, inputTraining, inputTrainingAns,inputTesting, inputValidate):
		#Initializes Logistic Regression Class
		#Preprocesses and sorts necessary data
		
		#save input data
		#self.setTraining = inputTraining
		#self.setTesting = inputTesting
		self.setAnswer = inputTrainingAns.reshape(inputTraining.shape[0],1)
		self.percentValidate = inputValidate
		self.setTraining, self.setTesting = self.feature_normalize(inputTraining,inputTesting)

		setTraining = fc.sort_ranges(self.setTraining)
		setTraining = fc.sort_data(self.setTraining)

		setTesting = fc.sort_ranges(self.setTesting)
		setTesting = fc.sort_data(self.setTesting)

		mu = -1.0
		sigma = 1.5
		#Initialize Number of Features,  Weights, Bias
		self.numWeights = (self.setTraining.shape[1])

		self.w = sigma* np.random.rand(self.numWeights,1) + mu 
		self.b = sigma * np.random.rand(1,1) + mu

		#self.setTraining = self.normalize_range(self.setTraining)
		#self.setTraining, meanI, stDevI = self.normalize_mean(self.setTraining)
		#self.meanTraining = meanI
		#self.stDevTraining = stDevI

		self.setTraining = np.append(self.setTraining,self.setAnswer,1)
		#Shuffle data
		np.random.seed(1)
		np.random.shuffle(self.setTraining)

		#Get total number of available data points
		numData = self.setTraining.shape[0]
		#Get Index to segment data
		idxSegment = numData - int(numData*self.percentValidate)
		#print (idxSegment)
		#print(idxSegment)
		print(idxSegment)
		#Segments total training data into validation and training data sets
		self.setValidation = self.setTraining[idxSegment:,:]
		self.setTraining = self.setTraining[:idxSegment,:]
		#print(self.setValidation.shape)
		#print(self.setTraining.shape)
		print ("Initialize Complete!")

	def feature_normalize(self,X_train, X_test):
		# feature normalization with all X
		X_all = np.concatenate((X_train, X_test))
		mu = np.mean(X_all, axis=0)
		sigma = np.std(X_all, axis=0)
		
		# only apply normalization on continuos attribute
		index = [0, 1, 3, 4, 5]
		mean_vec = np.zeros(X_all.shape[1])
		std_vec = np.ones(X_all.shape[1])
		mean_vec[index] = mu[index]
		std_vec[index] = sigma[index]

		X_all_normed = (X_all - mean_vec) / std_vec

		# split train, test again
		X_train_normed = X_all_normed[0:X_train.shape[0]]
		X_test_normed = X_all_normed[X_train.shape[0]:]

		return X_train_normed, X_test_normed


	def normalize_mean(self,setInput):
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

	def sigmoid(self,X):
		'''Compute the sigmoid function '''
		#d = zeros(shape=(X.shape))
		den = 1.0 + np.exp(-1.0 * X)
		output = 1.0 / den
		if np.isnan(output).any():
			print("NAN FOUND NAN FOUND NAN FOUND NAN FOUND NAN FOUND ")
			return -1
		#check for Nan?
		return np.clip(output,0.000000000000000001,0.99999999999999999999999)

	def compute_cost(self,X,y): 
	#computes cost given predicted and actual values
		#loss = (-y)*np.log(X) - (1-y)*np.log(1-X)
		loss = (1.0*np.sum(np.absolute(X-y)))/y.size
		return 1-loss

	def bound_prediction(self,prediction):
		boundary = np.mean(prediction)
		#print("Boundary: " + str(boundary))
		'''
		ans = prediction
		for i in range(len(ans)):
			if ans[i] >= (1.0-boundary):
				ans[i] = 1
			else:
				ans[i] = 0
		'''
		#print(prediction[0:5])
		ans = np.rint(prediction)
		#print(ans[0:5])
		return ans
	def train_logistic(self,iteration, eta,lambdaC):
		#normalize??
		inputData = self.setTraining[:,:-1]
		inputAns = self.setTraining[:,-1]
		inputAns = inputAns.reshape(inputAns.shape[0],1)

		inputValidation = self.setValidation[:,:-1]

		inputValidAns = self.setValidation[:,-1]

		inputValidAns = inputValidAns.reshape(inputValidation.shape[0],1)

		dwTotal = 0
		dbTotal = 0

		for idx in range(iteration):
		
			z = np.dot(inputData,self.w) + self.b
			zValid = np.dot(inputValidation,self.w) + self.b

			prediction = self.sigmoid(z)
			predictionValid = self.sigmoid(zValid)

			tmpWeight = np.dot(-(inputAns - prediction).T,inputData)
			tmpBias = np.sum(-(inputAns-prediction))

			dwTotal += tmpWeight ** 2
			dbTotal += tmpBias ** 2

			self.w -= ((eta*tmpWeight.T)+((lambdaC/2)*dwTotal.T))/np.sqrt(dwTotal).T
			self.b -= ((eta*tmpBias)+((lambdaC/2)*dbTotal))/np.sqrt(dbTotal)

			ansTraining = self.bound_prediction(prediction)
			ansValid = self.bound_prediction(predictionValid)
			#update weights
			#self.weights += (eta * diffWeight)/np.sqrt(dwTotal) #- (2 * coeff * self.weights)
			#self.bias += (eta * diffBias)/np.sqrt(dbTotal)

			if (idx % 100) == 0:
				loss = self.compute_cost(ansTraining,inputAns)
				lossValid = self.compute_cost(ansValid, inputValidAns)
				print ("It: %d  Train Acc: %f Valid Acc: %f" \
					% (idx,loss,lossValid))
				#print((lambdaC/2)*dwTotal.T)
		print("training done!")
		return 0

	def run_log_model(self):
		#Runs test set on trained weights
		setTesting = self.setTesting

		z = np.dot(setTesting,self.w) + self.b
		prediction = self.sigmoid(z)
		finalPrediction = self.bound_prediction(prediction)
		#initialize variables to save to CSV files
		csvOutput = np.zeros((16281+1,1+1), dtype ="|S6")
		#append labels and headers
		csvOutput[0,0] = "id"
		csvOutput[0,1] = "label"

		#append data
		for idx in range (16281):
			csvOutput[idx+1,0] = str(idx+1)
			csvOutput[idx+1,1] = int(finalPrediction[idx,0])
		timestamp = time.strftime("%Y%m%d-%H%M%S")
		filename = "./results/log_output_"+timestamp+".csv"
		weightName = "./results/log_weight_"+timestamp+".csv"
		biasName = "./results/log_bias_"+timestamp+".csv"
		#Write data to CSV
		np.savetxt(weightName, self.w, delimiter = ",", fmt = "%s")
		np.savetxt(biasName, self.b, delimiter = ",", fmt = "%s")
		np.savetxt(filename, csvOutput, delimiter=",", fmt = "%s")
		#np.savetxt(sys.argv[3], csvOutput, delimiter=",", fmt = "%s")

		print("Save Complete!")
		return 0

