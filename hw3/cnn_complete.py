from scipy import ndimage
import numpy as np
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
from keras.optimizers import SGD, adam
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import CSVLogger,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class ConvNet():
	#This class contains functions to perform Gradient
	def __init__(self, inputTrainName, inputTestName,augmentFlag,zmuvFlag):
		#Initializes Logistic Regression Class
		#Preprocesses and sorts necessary data
		self.fileTraining = inputTrainName
		self.fileTesting = inputTestName
		'''
		print("Reading Data...")
		yTrain  = np.genfromtxt(inputTrainName, dtype=int, skip_header=True, delimiter = ",", usecols = 0)
		yTrain = yTrain.reshape(yTrain.shape[0],1)
		tmpTrainData = np.genfromtxt(inputTrainName,dtype=str,skip_header=True,delimiter =",",usecols = 1)

		print("Sorting Data...")
		tmpTrainData = tmpTrainData.tolist()
		xTrain = []
		for i in range(len(tmpTrainData)):
			tmpRow = map(int,tmpTrainData[i].split(' '))
			xTrain.append(tmpRow)
		xTrain = np.asarray(xTrain)

		self.xTrain = xTrain.astype(np.float32)
		self.yTrain = yTrain.astype(np.int)#(28709,1)
		'''
		with open(sys.argv[1]) as trainFile:
		 	trainList = trainFile.read().splitlines()
		 	train_arr = np.array([line.split(",") for line in trainList])
			x_arr = train_arr[1:,1]
			y_arr = train_arr[1:,0]
			x_arr = np.array([str(line).split() for line in x_arr])
		  	y_arr = np.array([str(line).split() for line in y_arr])

	  	self.xTrain = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
	  	self.yTrain = y_arr.astype(np.int)#(28709,1)	

		#self.xTrain = self.xTrain.astype(np.float32)
		#self.yTrain = self.yTrain.astype(np.int)#(28709,1)
		self.remove_noise()
		
		if augmentFlag == 1: self.augment_data()

		self.normalize()

		if zmuvFlag == 1: self.ZMUV()

		self.model_prep()

		print("Initialization Complete")

	def print_image(self):
		'''
		xTrain = xTrain.reshape(xTrain.shape[0], 48, 48)
		print(xTrain.shape)
		plt.figure(1)
		plot1 = plt.imshow(xTrain[0])
		plt.figure(2)
		tmpTrain = np.fliplr(xTrain[0])
		plot2 = plt.imshow(tmpTrain)
		plt.show()
		xTrain = xTrain.reshape(xTrain.shape[0], 48*48)
		'''
		return 0

	def remove_noise(self):
		#print(xTrain.shape)
		xTrainMean = np.mean(self.xTrain,axis=1)
		idxZero = np.where(xTrainMean==0)[0]
		self.xTrain = np.delete(self.xTrain,idxZero,axis=0)
		self.yTrain = np.delete(self.yTrain,idxZero)

	def augment_data(self):
		#xTrain = np.hstack((yTrain,xTrain))
		print("Augmenting Data...")
		#Consider rotating 90 degrees here and reshaping back to 1D???
		self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], 48, 48).astype(np.float32)
		tmpTrain = np.fliplr(self.xTrain)
		self.xTrain = np.vstack((self.xTrain,tmpTrain))
		self.yTrain = np.concatenate((self.yTrain,self.yTrain))

	def normalize(self):
		self.xTrain /= 255

	def ZMUV(self):
		#Zero Mean Unity Variance
		print("Zero Mean and Unity Variance Processing...")
		xTrain = self.xTrain
		xTrainMean = np.mean(xTrain,axis=1)
		xTrainStd = np.std(xTrain,axis=1)

		for i in range(xTrain.shape[0]):
			for j in range(xTrain.shape[1]):
				try:
					xTrain[i,j] = (xTrain[i,j]-xTrainMean[i]) / xTrainStd[i]
				except:
					#print("idx:")
					#print(xTrain[i,j],xTrainMean[i],xTrainStd[i])
					print('Invalid Type of NaN found!')
		self.xTrain = xTrain

	def model_prep(self):

		#self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], 48, 48, 1).astype(np.float32)

		#convert class vectors to binary class matrices (one hot vectors)
		self.original, self.idx = np.unique(self.yTrain, return_inverse = True)
		self.yTrain = np_utils.to_categorical(self.yTrain, 7)
		print("Data Prepared!")
	def model_generate(self):
		self.model = Sequential()
		print("Empty Model Generated")

	def init_conv2D_layer(self,layerNum,typeAct,poolNum,dropNum,alpha):
			self.model.add(Conv2D(layerNum,(3,3) ,padding="valid", input_shape = (48,48,1)))
			self.model.add(BatchNormalization())
			
			if typeAct==0:
				self.model.add(Activation("relu"))
			elif (typeAct ==1):
				self.model.add(LeakyReLU(alpha=alpha))
			else:
				print("Activation Model Not Specified!")
			self.model.add(MaxPooling2D(poolNum,poolNum))
			self.model.add(Dropout(dropNum))
			print("Conv2D Layer Added")

	def add_conv2D_layer(self,layerNum,typeAct,poolNum,dropNum,alpha):
		self.model.add(Conv2D(layerNum,(3,3)))
		self.model.add(BatchNormalization())
		
		if (typeAct==0):
			self.model.add(Activation("relu"))
		elif (typeAct ==1):
			self.model.add(LeakyReLU(alpha=alpha))
		else:
			print("Activation Model Not Specified!")
		self.model.add(MaxPooling2D(poolNum,poolNum))
		self.model.add(Dropout(dropNum))
		print("Conv2D Layer Added")

	def add_dense_layer(self, layerNum, dropNum):
		self.model.add(Dense(layerNum, kernel_initializer='normal', \
						#kernel_regularizer = regularizers.l2(0.01),
						activation='relu'))
		self.model.add(Dropout(dropNum))
		print("Dense Layer Added")

	def add_softmax_layer(self,layerNum):
		self.model.add(Dense(layerNum, kernel_initializer='normal', activation='softmax'))

	def flatten_model(self):
		self.model.add(Flatten())

	def run_model(self,batch_size,nb_epoch,validation,lr, method):
		print("Running Model")

		if (method == 1):
			self.datagen = ImageDataGenerator(
					    featurewise_center=True,\
					    featurewise_std_normalization=True,\
						samplewise_center=False,\
					    samplewise_std_normalization=False,\
					    rotation_range=20,\
					    width_shift_range=0.2,\
					    height_shift_range=0.2,\
					    zca_whitening=True,\
					    horizontal_flip=True) 
			print ("Fitting Generator")
			self.datagen.fit(self.xTrain)


		print ("Compiling Model")

		if (method == 2):
			print("SGD Selected")
			sgd = SGD(lr=lr, decay=1e-7, momentum=0.7, nesterov=True)
			self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		
		else:
			self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


		self.timestamp = time.strftime("%Y%m%d-%H%M%S")
		logName = "./results/"+self.timestamp+"_log.log"
		csv_logger = CSVLogger(logName)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0008)
		#callback_list = [csv_logger,reduce_lr]
		print ("Fitting Model")

		if (method == 0 or method == 2): #regular fitting
			callback_list = [csv_logger,reduce_lr]
			self.history = self.model.fit(self.xTrain, self.yTrain, batch_size=batch_size, epochs=nb_epoch, verbose=1, \
								shuffle=True, validation_split=validation,callbacks=callback_list)
		elif (method == 1): #batch fitting
			callback_list = [csv_logger]
			self.history = self.model.fit_generator(self.datagen.flow(self.xTrain,\
										self.yTrain), \
										steps_per_epoch=32,\
										epochs = nb_epoch,\
										verbose = 1,\
										callbacks=callback_list)
			#(len(self.xTrain) / batch_size),\
		#steps_per_epoch=batch_size,\
		#show_accuracy=True,\
		else:
			print("Method Not Selected")
								


		print("Saving Log File")
		modelName = "./results/"+self.timestamp+"_model.h5"
		self.model.save(modelName)

	def predict_test_data(self,batch_size):
		print("Reading Test Data...")

		with open(sys.argv[2]) as testFile:
			testList = testFile.read().splitlines()
			test_arr = np.array([line.split(",") for line in testList])
		  	test_x_arr = test_arr[1:,1]
		  	test_x_arr = np.array([str(line).split() for line in test_x_arr])

		  	xTest = test_x_arr.reshape(test_x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

		xTest /= 255

		yTest  = np.genfromtxt(self.fileTesting, dtype=int, skip_header=True, delimiter = ",", usecols = 0)
		yTest = yTest.reshape(yTest.shape[0],1)

		'''
		tmpTestData = np.genfromtxt(self.fileTesting,dtype=str,skip_header=True,delimiter =",",usecols = 1)

		print("Sorting Test Data...")
		tmpTestData = tmpTestData.tolist()
		xTest = []
		for i in range(len(tmpTestData)):
			tmpRow = map(int,tmpTestData[i].split(' '))
			xTest.append(tmpRow)
		xTest = np.asarray(xTest)

		print("Preprocessing Test Data...")
		#rescale
		xTest /= 255
		'''
		xTest = xTest.reshape(xTest.shape[0], 48, 48, 1).astype(np.float32)

		print("Predicting...")
		#yTestPredict = self.model.predict(xTest)
		#yTest = np.around(yTestPredict)
		#yTest = self.original[yTest.argmax(1)]

		yTest = self.model.predict_classes(xTest, batch_size = batch_size)

		print("Saving Prediction to CSV...")
		outputTest = np.zeros((len(yTest)+1, 2), dtype='|S5')
		outputTest[0,0] = "id"
		outputTest[0,1] = "label"

		for i in range (outputTest.shape[0]-1):
		    outputTest[i+1,0] = str(i)
		    outputTest[i+1,1] = str(yTest[i])

		origName = "./results/"+self.timestamp+"_original"
		filename = "./results/"+self.timestamp+"_prediction.csv"

		np.savetxt(origName, self.original, delimiter=",", fmt="%s")
		np.savetxt(filename, outputTest, delimiter=",", fmt="%s")
		#np.savetxt(sys.argv[3], test_output, delimiter=",", fmt = "%s")

		print("complete!")
