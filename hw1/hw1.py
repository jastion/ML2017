import sys
import numpy as np
import matplotlib as pl
import time
import csv
from lin_grad import LineGradDesc

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
csvTraining  = np.genfromtxt(sys.argv[1], dtype="f", skip_header=False, delimiter = ",")
csvTesting = np.genfromtxt(sys.argv[2], skip_header=True, delimiter = ",")

csvTraining = csvTraining[1:,2:]
#test = csvTraining[1:5,2:8]
#print (test)
#test = np.split(test,2)
#print (test)
#test = np.hstack(test)
#print (test)
#print (test.shape)
#print (test)
#features = np.array([9])
features = np.arange(18)
#print (features)
hours = np.arange(24) 
print(features)
print(hours)
#print (test)
lineGrad = LineGradDesc(csvTraining, csvTesting , features, hours, 0.2)


csvOutput = csvTraining
filename = time.strftime("%Y%m%d-%H%M%S_output.csv")
#np.savetxt(filename, csv_output,fmt="%s", delimiter=","), 