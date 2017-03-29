class LogDesc:
	#This class contains functions to perform Gradient
	def __init__(self, inputTraining, inputTrainingAns,inputTesting, inputValidate,):
		#Initializes Logistic Regression Class
		#Preprocesses and sorts necessary data
		
		#save input data
		self.setTraining = inputTraining
		self.setTesting = inputTesting
		self.percentValidate = inputValidate

		mu = -1
		sigma = 0.5
		#Initialize Number of Features,  Weights, Bias
		self.numWeights = len(self.inputTraining)

		self.weights = sigma* np.random.rand(self.numWeights,1) + mu 
		self.bias = sigma * np.random.rand(1,1) + mu

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

	def sigmoid(self,X):
	    '''Compute the sigmoid function '''
	    #d = zeros(shape=(X.shape))

	    den = 1.0 + e ** (-1.0 * X)
	    output = 1.0 / den
		return output

	def compute_cost(self,theta,X,y): 
	#computes cost given predicted and actual values
	    m = X.shape[0] #number of training examples
	    theta = reshape(theta,(len(theta),1))

	    #y = reshape(y,(len(y),1))
	    
	    J = (1./m) * (-transpose(y).dot(log(self.sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-self.sigmoid(X.dot(theta)))))
	    
	    grad = transpose((1./m)*transpose(self.sigmoid(X.dot(theta)) - y).dot(X))
	    #optimize.fmin expects a single value, so cannot return grad
		return J[0][0]#,grad

	def logDesc(self):

