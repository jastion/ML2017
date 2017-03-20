import sys
import numpy as np
import matplotlib as mpl
import time
import csv
from lin_grad import LineGradDesc
#from GradientDescent import GradDesc

# 0  AMB_TEMP		9	PM2.5  
# 1  CH4			10	RAINFALL
# 2  CO 			11	RH (Rel. Humidity)
# 3  NMHC			12	SO2
# 4  NO 			13 	THC
# 5  NO2 			14 	WD_HR
# 6  NOx 			15	WIND_DIRECT
# 7  O3				16	WIND_SPEED
# 8  PM10			17 	WS_HR


np.set_printoptions(linewidth=1e3, edgeitems=1e2, suppress=True,precision=3)

csvTraining  = np.genfromtxt(sys.argv[1], dtype="S", skip_header=True, delimiter = ",")
csvTraining = csvTraining[:,3:] 
csvTesting = np.genfromtxt(sys.argv[2], dtype="S", skip_header=False, delimiter = ",")
csvTesting = csvTesting[:,2:]


setTrain = csvTraining[:18,:]

for days in range(1,12*20):
    setTrain = np.append(setTrain, csvTraining[days*18:days*18+18,:],1)

setTrain[setTrain == "NR"] = 0#(if array have "NR" string, let it convert to float)
setTrain = setTrain.astype(np.float)#(18,5760)

setTest = csvTesting[:18,:]

for days in range(1,12*20):
    setTest = np.append(setTest, csvTesting[days*18:days*18+18,:],1)

setTest[setTest == "NR"] = 0
setTest = setTest.astype(np.float)#(18,2160)

features = np.arange(18)
#features = np.array([8,9,10,16]) 5.72 5.9, 6.19
#features = np.array([8,9,10,16]) #5.89221 5,86 6.17
features = np.array([4,5,6,8,9,10,16])
#features = np.array([9,10,16])
features = np.array([9])
hours = np.arange(9)
print ("Initializing")

lineGrad = LineGradDesc(setTrain, setTest , features, hours, 0.2) 
#lineGrad.grad_desc(100000, 0.5)
#lineGrad.run_test_set()
lineGrad.neural_network(5000,0.1)
#lineGrad.test_nn()
#predict = GradDesc(csvTraining, csvTesting, valid_percent = 0.2)
##predict.train_grad_ada(interation = 200, lr_rate = 0.2)
#predict.test_function()
#lineGrad.run_test_set()
filename = time.strftime("%Y%m%d-%H%M%S_output.csv")
#np.savetxt(filename, csv_output,fmt="%s", delimiter=","), 
