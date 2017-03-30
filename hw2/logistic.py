import numpy as np
import feature_classification as fc
import sys
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

		mu = -1
		sigma = 0.5
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
		np.random.shuffle(self.setTraining)

		#Get total number of available data points
		numData = self.setTraining.shape[0]
		#Get Index to segment data
		idxSegment = numData - int(numData*self.percentValidate)
		#print (idxSegment)
		#print(idxSegment)
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
		#check for Nan?
		return np.clip(output,0.000000000000000001,0.99999999999999999999999)

	def compute_cost(self,X,y): 
	#computes cost given predicted and actual values
		#loss = (-y)*np.log(X) - (1-y)*np.log(1-X)
		loss = (1.0*np.sum(np.absolute(X-y)))/y.size
		return 1-loss

	def bound_prediction(self,prediction):
		boundary = np.mean(prediction)
		#print(prediction[0:5])
		ans = np.rint(prediction)
		#print(ans[0:5])
		return ans
	def train_logistic(self,iteration, eta):
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

			self.w -= (eta*tmpWeight.T)/np.sqrt(dwTotal).T
			self.b -= (eta*tmpBias)/np.sqrt(dbTotal)

			ansTraining = self.bound_prediction(prediction)
			ansValid = self.bound_prediction(predictionValid)
			#update weights
			#self.weights += (eta * diffWeight)/np.sqrt(dwTotal) #- (2 * coeff * self.weights)
			#self.bias += (eta * diffBias)/np.sqrt(dbTotal)

			if (idx % 1000) == 0:
				loss = self.compute_cost(ansTraining,inputAns)
				lossValid = self.compute_cost(ansValid, inputValidAns)
				print ("It: %d  Train Acc: %f Valid Acc: %f" \
					% (idx,loss,lossValid))

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
		#Write data to CSV
		np.savetxt("w_pm25.csv", self.w, delimiter = ",", fmt = "%s")
		np.savetxt("b_pm25.csv", self.b, delimiter = ",", fmt = "%s")
		np.savetxt("prediction.csv", csvOutput, delimiter=",", fmt = "%s")
		#np.savetxt(sys.argv[3], csvOutput, delimiter=",", fmt = "%s")

		print("Save Complete!")
		return 0

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
		hiddenLayer = 50
		#initialize synapses
		syn0 = 2*np.random.random((self.numWeights,self.numWeights)) - 1
		#syn0 = 2*np.random.random((self.numWeights,1)) - 1
		#syn1 = 2*np.random.random((hiddenLayer,1)) - 1	

		#initialize previous weights
		prevSynWeightUpdate0 = np.zeros_like(syn0)
		#prevSynWeightUpdate1 = np.zeros_like(syn1)

		syn0DirCount = np.zeros_like(syn0)
		#syn1DirCount = np.zeros_like(syn1)
		#alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]
		alphas = [0.0001]
		for alpha in alphas:
			#print (alpha)
			#Iterate over neural network
			for idx in xrange(iterations):
				layer0 = X
				layer0V = xValid

				layer1 = self.sigmoid(np.dot(layer0,syn0))
				#layer2 = self.sigmoid(np.dot(layer1,syn1))

				layer1V = self.sigmoid(np.dot(layer0V,syn0))
				#layer2V = self.sigmoid(np.dot(layer1V,syn1))

				#errorLayer2V = yValid - layer2V
				#errorLayer2 = y - layer2
				errorLayer1 = y - layer1
				#deltaLayer2 = errorLayer2*self.sigmoid_output_to_derivative(layer2)
				#errorLayer1 = deltaLayer2.dot(syn1.T)
				deltaLayer1 = errorLayer1*self.sigmoid_output_to_derivative(layer1)

				
				errorLayer1V = yValid - layer1V
				if (idx % 100) == 0:
					#a = str(np.mean(np.abs(errorLayer2)))
					b = str(np.mean(np.abs(errorLayer1)))
					d = str(np.mean(np.abs(errorLayer1V)))
					#c = str(np.mean(np.abs(errorLayer2V)))
					print ("Iter:" + str(idx) + "  Error1:" + b + "    Validation: " + d)

				synWeightUpdate0 = layer1.T.dot(deltaLayer1)
				#synWeightUpdate0 = layer1.T.dot(deltaLayer2)
				#synWeightUpdate1 = layer2.T.dot(deltaLayer1)

				#if idx > 0:
			#		syn0DirCount += np.abs(((synWeightUpdate0 > 0)+0) - ((prevSynWeightUpdate0 > 0) + 0))
		#			syn1DirCount += np.abs(((synWeightUpdate1 > 0)+0) - ((prevSynWeightUpdate1 > 0) + 0))
				
				#syn1 -= eta * synWeightUpdate1.T
				#print(syn0.shape)
				#print(alpha)
				#print(synWeightUpdate0.shape)
				syn0 += alpha * synWeightUpdate0.T

				prevSynWeightUpdate0 = synWeightUpdate0
				#prevSynWeightUpdate1 = synWeightUpdate1

			#np.savetxt("./data/NN.csv", layer2, delimiter = ",", fmt = "%s")
		self.syn0 = syn0
		#self.syn1 = syn1
		return 0
	def sigmoid_output_to_derivative(self,output):
		#Sigmoid derivative function
		return output*(1-output)

	def test_nn(self):
		#Test function to implement Neural network
		setTesting = self.setTesting
		#prediction = setTesting.dot(self.weights)+self.bias

		syn0 = self.syn0
		#syn1 = self.syn1

		layer0 = setTesting
		layer1 = self.sigmoid(np.dot(layer0,syn0))
		#layer2 = self.sigmoid(np.dot(layer1,syn1))
		layer1 = self.bound_prediction(layer1)

		csvOutput = np.zeros((16281+1,1+1), dtype ="|S6")
		csvOutput[0,0] = "id"
		csvOutput[0,1] = "label"

		for idx in range (16281):
			csvOutput[idx+1,0] = str(idx+1)
			csvOutput[idx+1,1] = int(layer1[idx,0])

		np.savetxt("./data/syn0_pm25.csv", syn0, delimiter = ",", fmt = "%s")
		#np.savetxt("./data/syn1_pm25.csv", syn1, delimiter = ",", fmt = "%s")
		np.savetxt("./data/NN_output.csv", layer1, delimiter=",", fmt = "%s")
		return 0