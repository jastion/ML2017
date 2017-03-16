import sys
import numpy as np
import matplotlib as pl

print (sys.argv[0])  
print (sys.argv[1])
print (sys.argv[2])

header_training = np.genfromtxt(sys.argv[1], skip_header=True, delimiter = ",",names=True)
csv_training  = np.genfromtxt(sys.argv[1], skip_header=True, delimiter = ",",names=True)
csv_testing = np.genfromtxt(sys.argv[2], skip_header=True, delimiter = ",")


print (csv_training)
print (csv_training.shape)
print (header_training)
