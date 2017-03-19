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
csvTraining  = np.genfromtxt(sys.argv[1], dtype="f", skip_header=False, delimiter = ",")
csvTesting = np.genfromtxt(sys.argv[2], dtype="f", skip_header=False, delimiter = ",")

csvTraining = csvTraining[1:,2:]
csvTesting = csvTesting[:,2:]
features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
hours = np.arange(9)
lineGrad = LineGradDesc(csvTraining, csvTesting , features, hours, 0.2, 2, 9)
lineGrad.grad_desc(1001, 1.0)
lineGrad.run_test_set()
#predict = GradDesc(csvTraining, csvTesting, valid_percent = 0.2)
##predict.train_grad_ada(interation = 200, lr_rate = 0.2)
#predict.test_function()
#lineGrad.run_test_set()
filename = time.strftime("%Y%m%d-%H%M%S_output.csv")
#np.savetxt(filename, csv_output,fmt="%s", delimiter=","), 