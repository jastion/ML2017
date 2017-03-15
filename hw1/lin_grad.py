import sys
import numpy as py
import matplotlib as pl

print (sys.argv[0])  
print (sys.argv[1])
print (sys.argv[2])
csv_training  = np.genfromtxt(sys.argv[1], delimiter=',')
csv_testing = np.genfromtext(sys.argv[2],delimiter=',')


