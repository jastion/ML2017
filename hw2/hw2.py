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

""
>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

https://archive.ics.uci.edu/ml/datasets/Adult
http://www.mis.nsysu.edu.tw/db-book/DMProject2007Spring/6/project.pdf
https://github.com/CommerceDataService/tutorial-predicting-income/blob/master/predicting_income_with_census_data_pt1.md
http://scg.sdsu.edu/dataset-adult_r/
'''

'''
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
'''
age,  O
education, O 
marital-status, O
occupation O
capital-gain, O?
hours-per-week, O?
'''

import pandas as pd 
import numpy as np
import sys
import feature_classification as fc
import logistic as log

#marital-status, O
#occupation O
print(sys.argv)
#X Train
setTraining  = np.genfromtxt(sys.argv[3], dtype="float", skip_header=True, delimiter = ",")
#X_Test
setTesting  = np.genfromtxt(sys.argv[5], dtype="float", skip_header=True, delimiter = ",")



#setTraining = fc.cut_data(setTraining)
#setTesting = fc.cut_data(setTesting)

setAns = np.genfromtxt(sys.argv[4], dtype="float", skip_header=False, delimiter = ",")
setAns = setAns.reshape(setAns.shape[0],1)
#setTraining = fc.sort_capital(setTraining)
#setTesting = fc.sort_capital(setTesting)
#for i in range(3):
test = log.LogDesc(setTraining,setAns, setTesting,0.05)
test.train_logistic(1001,0.2)
test.run_log_model()

#lineGrad = LineGradDesc(setTraining, setTesting, 0.20) 


#np.savetxt("new_testing_data.csv", setTesting, delimiter = ",", fmt = "%s")
#np.savetxt("new_training_data.csv", setTraining, delimiter = ",", fmt = "%s")
#np.savetxt(sys.argv[6], setTraining, delimiter = ",", fmt = "%s")



#dfCsv = (dfCsv[dfCsv=='\?'])
#print(dfCsv.shape)
#print(dfCsv)
#print(idx)
#.to_csv("ztest_output.csv",sep=',')
#print ~((df['X'] == '?' )  (df['Y'] == '?' ) | (df['Z'] == '?' ))
