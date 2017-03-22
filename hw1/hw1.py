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

from lin_grad import LineGradDesc
#from GradientDescent import GradDesc

# 0  X	AMB_TEMP		9	O	PM2.5  
# 1  	CH4				10	O	RAINFALL
# 2  	CO 				11	X	RH (Rel. Humidity)
# 3  	NMHC			12	X	SO2
# 4  X	NO 				13 		THC
# 5  	NO2 			14 	X	WD_HR
# 6  	NOx 			15	O	WIND_DIRECT
# 7  	O3				16	O	WIND_SPEED
# 8  O	PM10			17 		WS_HR


np.set_printoptions(linewidth=1e3, edgeitems=1e2, suppress=True,precision=3)

#Read in Training Data and preprocess
csvTraining  = np.genfromtxt(sys.argv[1], dtype="S", skip_header=True, delimiter = ",")
csvTraining = csvTraining[:,3:] 
setTrain = csvTraining[:18,:]

for days in range(1,12*20):
    setTrain = np.append(setTrain, csvTraining[days*18:days*18+18,:],1)

#Read in Testing data and preprocess
csvTesting = np.genfromtxt(sys.argv[2], dtype="S", skip_header=False, delimiter = ",")
csvTesting = csvTesting[:,2:]

setTrain[setTrain == "NR"] = 0#(if array have "NR" string, let it convert to float)
setTrain = setTrain.astype(np.float)#(18,5760)

setTest = csvTesting[:18,:]

for days in range(1,12*20):
    setTest = np.append(setTest, csvTesting[days*18:days*18+18,:],1)

setTest[setTest == "NR"] = 0
setTest = setTest.astype(np.float)#(18,2160)

#Feature selection and time range


#features = np.array([8,9,10,16]) 5.72 5.9, 6.19
#features = np.array([8,9,10,16]) #5.89221 5,86 6.17
#features = np.array([4,5,6,8,9,10,16])
#features = np.array([8,9,10,16]) #5.78
#features = np.array([4,8,9,10,16])

feature1 = np.array([9])
feature4 = np.array([8,10,12,16])
feature6 = np.array([8,9,10,14,15,16])
feature10 = np.array([0,4,5,7,8,10,12,14,15,16])
featureAll = np.arange(18)
featureIndex = [feature1,featureAll,feature4,feature6,feature10]

hours = np.array([0,1,2,3,4,5,6,7,8])
hour1 = np.array([0,1,2,3,])
hour2 = np.array([0,1,2,3,4])
hour3 = np.array([0,1,2,3,4,5])
hour4 = np.array([0,1,2,3,4,5,6])
hour5 = np.array([0,2,3,4,5,6,7,8])

hourIndex = [hour1,hour2,hour3,hour4,hour5]
print ("Initializing")
#iterations = [100,500]
lr = [0.001,0.01,0.1,1,10,100]
markerIndex = ['s', 'o','v','p','d']
colorIndex = ['r','b','g','k','c']
learningIndex = [0.01,0.1,1,10,100]
orderIndex = [1,2,3]
orderValid = [0.1,0.2,0.4,0.6,0.8]
regIndex = [0,0.001,0.01,0.1,1] 
#Run Gradient Descent
#mpl.figure(1)
#print(len(featureIndex))
#lineGrad = LineGradDesc(setTrain, setTest , feature6, hours, 0.50) 
#errorValid, errorTraining = lineGrad.grad_desc(, 0.5)

for idx in range(len(featureIndex)):
	#idxIteration = iterations[idx]
	idxIteration = 5000
	idxMarker = markerIndex[idx]
	idxColor = colorIndex[idx]
	
	idxLR = learningIndex[idx]

	idxFeature = featureIndex[0]
	idxOrder = orderIndex[0]
	idxValid = orderValid[idx]
	idxReg = regIndex[idx]

	print ("Validation %" + str(idxValid))
	lineGrad = LineGradDesc(setTrain, setTest , idxFeature, hours, 0.2,idxOrder,idxReg) 
	errorValid, errorTraining = lineGrad.grad_desc(idxIteration, 0.2)
	arrayIter = np.arange(idxIteration)

	idxLabel = "Lamba: " + str(idxReg)
	#idxLabel = "Data " + str((1.0-idxValid)*100) + "%"

	print(arrayIter.shape)
	print(errorValid.shape)
	mpl.figure(1)
	mpl.plot(arrayIter, errorValid, color = idxColor,label=idxLabel)
	mpl.title('Validation Error vs Iterations', fontsize=20)
	mpl.xlabel('Iterations', fontsize=18)
	mpl.ylabel('Validation Error', fontsize=16)
	mpl.legend()

	mpl.figure(2)
	mpl.plot(arrayIter, errorTraining, color = idxColor, label=idxLabel)
	mpl.title('Training Error vs Iterations', fontsize=20)
	mpl.xlabel('Iterations', fontsize=18)
	mpl.ylabel('Training Error', fontsize=16)
	mpl.legend()

	diffError = errorValid - errorTraining
	mpl.figure(3)
	mpl.plot(arrayIter, diffError, color = idxColor, label=idxLabel)
	mpl.title('Difference in Training and Validation Error', fontsize=20)
	mpl.xlabel('Iterations', fontsize=18)
	mpl.ylabel('Error', fontsize=16)
	mpl.legend()
mpl.show()


lineGrad.run_test_set()
#Not implemented
#lineGrad.random()
#lineGrad.neural_network(2000,0.1)
#lineGrad.test_nn()