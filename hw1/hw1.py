# Machine Learning 2017
# Homework 1: 	Linear Regression
# Instructor: 	李宏毅 (Hung-yi Lee)
# Student: 		邱名彦 / Michael Chiou
# Student ID: 	R05921086
# Email:		r05921086.ntu.edu.tw 
# Github: 		https://github.com/jastion/ML2017.git

import sys
import numpy as np
import matplotlib as mpl
import time
from lin_grad import LineGradDesc
#from GradientDescent import GradDesc

# 0  	AMB_TEMP		9	O	PM2.5  
# 1  	CH4				10	O	RAINFALL
# 2  	CO 				11	X	RH (Rel. Humidity)
# 3  	NMHC			12	X	SO2
# 4  X	NO 				13 		THC
# 5  	NO2 			14 		WD_HR
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
#features = np.array([9])
#features = np.arange(18)
#features = np.array([8,9,10,16]) 5.72 5.9, 6.19
#features = np.array([8,9,10,16]) #5.89221 5,86 6.17
#features = np.array([4,5,6,8,9,10,16])
#features = np.array([8,9,10,16]) #5.78
#features = np.array([4,8,9,10,16])
#features = np.array([9,10,12,16])
features = np.array([8,9,10,15,16])
hours = np.arange(9)
print ("Initializing")

#Run Gradient Descent
lineGrad = LineGradDesc(setTrain, setTest , features, hours, 0.2) 
lineGrad.grad_desc(100000, 1000)
lineGrad.run_test_set()

#Not implemented
#lineGrad.random()
#lineGrad.neural_network(2000,0.1)
#lineGrad.test_nn()