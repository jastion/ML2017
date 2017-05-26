import matplotlib.pyplot as plt
import numpy as np 
import sys


xAxesLength = 300
titlePlot = 'RNN w/ Word Embedding vs BagofWords F1 Score Comparison'
titleRNN = 'RNN'
titleCompare = 'BagofWords'
pngName = 'fig_bag.png'
#dataSigmoid = np.genfromtxt('training_sigmoid.log',delimiter = ',',dtype=str,skip_header=1)
#dataCompare = np.genfromtxt('training_bagofwords.log',delimiter = ',',dtype=str,skip_header=1)


#enter name of log files
dataSigmoid = np.genfromtxt(sys.argv[1],delimiter = ',',dtype=str,skip_header=1)
dataCompare = np.genfromtxt(sys.argv[2],delimiter = ',',dtype=str,skip_header=1)

idxSigmoid = np.where(dataSigmoid=='nan')
idxCompare = np.where(dataCompare=='nan')


dataSigmoid[idxSigmoid] = '0'
dataCompare[idxCompare] = '0'

dataSigmoid = dataSigmoid[:xAxesLength,3]
dataCompare = dataCompare[:xAxesLength,3]

dataSigmoid = dataSigmoid.astype(float) * 100.0
dataCompare = dataCompare.astype(float) * 100.0

x = np.arange(len(dataSigmoid))

fig = plt.figure()
plt.title(titlePlot)

plt.plot(x,dataSigmoid, label=titleRNN)
plt.plot(x,dataCompare, label=titleCompare)
plt.legend(loc = 'upper right')
plt.grid('on')


plt.axis([0, 55, 0, 70])

plt.xlabel('Epoch')
plt.ylabel('F1 Score')
fig.savefig(pngName, bbox_inches='tight', pad_inches=0)
#plt.show()
