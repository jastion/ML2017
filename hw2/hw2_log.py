## Machine Learning 2017
# Homework 2: 	
# Instructor: 	Hung-yi Lee
# Student: 		Michael Chiou
# Student ID: 	R05921086
# Email:		r05921086.ntu.edu.tw 
# Github: 		https://github.com/jastion/ML2017.git
# Python 2.7

#https://inclass.kaggle.com/c/ml2017-hw2
'''
age,  O
workclass, 
fnlwgt, 
education, O 
education num, 
marital-status, O
occupation O
relationship, 
race,
sex,
capital-gain, O?
capital-loss, 
hours-per-week, O?
native-country, 
make over 50K a year or not

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]
'''

import numpy as np
import sys
import feature_classification as fc
import logistic as log
#import matplotlib.pyplot as plt

print(sys.argv)
#X Train
setTraining  = np.genfromtxt(sys.argv[3], dtype="float", skip_header=True, delimiter = ",")
#X_Test
setTesting  = np.genfromtxt(sys.argv[5], dtype="float", skip_header=True, delimiter = ",")

setAns = np.genfromtxt(sys.argv[4], dtype="float", skip_header=False, delimiter = ",")
setAns = setAns.reshape(setAns.shape[0],1)

#setTraining = fc.sort_ranges(setTraining)
        #setTraining = fc.sort_data(self.setTraining)
#setTesting = fc.sort_ranges(setTesting)
        #setTesting = fc.sort_data(self.setTesting)

setTraining = fc.sort_capital(setTraining)
setTesting = fc.sort_capital(setTesting)

test = log.LogDesc(setTraining,setAns, setTesting,0.2)#0.00000000005
lossTrain, lossValid = test.train_logistic(501,0.2,0.00000000000)
test.run_log_model()
