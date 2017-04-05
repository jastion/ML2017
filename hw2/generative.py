import numpy as np
import feature_classification as fc
import sys

class ProbGenDesc:
	#This class contains functions to perform Gradient
	def __init__(self, inputTraining, inputTrainingAns,inputTesting, inputValidate):
		#Initializes Logistic Regression Class
		#Preprocesses and sorts necessary data
		
		#save input data
		self.setAnswer = inputTrainingAns.reshape(inputTraining.shape[0],1)
		self.percentValidate = inputValidate
		self.setTraining, self.setTesting = self.feature_normalize(inputTraining,inputTesting)

		self.setTraining = fc.sort_ranges(self.setTraining)
		#self.setTraining,idxRow = fc.sort_data(self.setTraining,0)
		self.setTraining = np.append(self.setTraining,self.setAnswer,1)


		self.setTesting = fc.sort_ranges(self.setTesting)
		#self.setTesting = fc.sort_data(self.setTesting,1)

		#Initialize Number of Features,  Weights, Bias
		self.w, self.b = self.classification()

		#Shuffle data
		np.random.shuffle(self.setTraining)

		#Get total number of available data points
		numData = self.setTraining.shape[0]
		#Get Index to segment data
		idxSegment = numData - int(numData*self.percentValidate)

		#Segments total training data into validation and training data sets
		self.setValidation = self.setTraining[idxSegment:,:]
		self.setTraining = self.setTraining[:idxSegment,:]

		print ("Initialize Complete!")

	def classification(self):
		#Calculate weights and bias
		#get number of values
		binaryValNum1 = np.where(self.setTraining[:,-1] == 0)
		binaryValNum2 = np.where(self.setTraining[:,-1] == 1)

		#count number of binary class values
		N1 = binaryValNum1[0].size#(19807)
		N2 = binaryValNum2[0].size#(6241)

		#Compile values into relevant array
		binaryValData1 = np.array([]).reshape(0,int(self.setTraining[:,:-1].shape[1]))
		binaryValData2 = np.array([]).reshape(0,int(self.setTraining[:,:-1].shape[1]))
		
		for idx in binaryValNum1[0]:
			binaryValData1 = np.vstack((binaryValData1,self.setTraining[idx,:-1]))#(19807,106)

		for idx2 in binaryValNum2[0]:
			binaryValData2 = np.vstack((binaryValData2,self.setTraining[idx2,:-1]))#(6241,106)
		
		#calculate mean of each array for each column
		mean1 = np.array([]).reshape(0,binaryValData1.shape[1])
		mean2 = np.array([]).reshape(0,binaryValData2.shape[1])

		for idx in range(binaryValData1.shape[1]):
			m = np.mean(binaryValData1[:,idx])
			mean1 = np.append(mean1,m)

		for idx in range(binaryValData2.shape[1]):
			m = np.mean(binaryValData2[:,idx])
			mean2 = np.append(mean2,m)

		mean1 = mean1.reshape(binaryValData1.shape[1],1)#(106,1)
		mean2 = mean2.reshape(binaryValData2.shape[1],1)#(106,1)

		#calculate covariance
		covar1 = np.cov(binaryValData1, rowvar = False, bias = True)#(106,106)
		covar2 = np.cov(binaryValData2, rowvar = False, bias = True)#(106,106)
		covarTotal = ((N1*1.0)/(N1+N2))*covar1+((N2*1.0)/(N1+N2))*covar2#(106,106)

		#calculate inverse covariance
		covarInvTotal = np.linalg.inv(covarTotal)

		#calculate weights
		w = np.dot((mean1-mean2).T, covarInvTotal).T#(106,1)

		#calculate bias
		b1 = (-1.0/2)*np.dot(np.dot((mean1.T), covarInvTotal), mean1)
		b2 = (1.0/2)*np.dot(np.dot((mean2.T), covarInvTotal), mean2)
		b = b1+b2+np.log((N1*1.0)/N2)
		print("classification completed")
		return w,b

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

	def sigmoid(self,X):
		'''Compute the sigmoid function '''
		#d = zeros(shape=(X.shape))
		den = 1.0 + np.exp(-1.0 * X)
		output = 1.0 / den
		#check for Nan?
		return np.clip(output,0.000000000000000001,0.99999999999999999999999)

	def bound_prediction(self,prediction):
		ans = np.rint(prediction)
		ans = abs(ans - 1)
		return ans


	def run_gen_model(self):
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
		#np.savetxt("w_gen.csv", self.w, delimiter = ",", fmt = "%s")
		#np.savetxt("b_gen.csv", self.b, delimiter = ",", fmt = "%s")
		#np.savetxt("prediction.csv", csvOutput, delimiter=",", fmt = "%s")
		np.savetxt(sys.argv[6], csvOutput, delimiter=",", fmt = "%s")

		print("Save Complete!")
		return 0
