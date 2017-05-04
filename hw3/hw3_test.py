import sys
import numpy as np
from keras.models import load_model

batch_size = 64
model = load_model('./model/model.h5')

print("Reading Test Data...")

with open(sys.argv[1]) as testFile:
	testList = testFile.read().splitlines()
	test_arr = np.array([line.split(",") for line in testList])
  	test_x_arr = test_arr[1:,1]
  	test_x_arr = np.array([str(line).split() for line in test_x_arr])

  	xTest = test_x_arr.reshape(test_x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

xTest /= 255
xTest = xTest.reshape(xTest.shape[0], 48, 48, 1).astype(np.float32)

print("Predicting...")

yTest = model.predict_classes(xTest, batch_size = batch_size)

print("Saving Prediction to CSV...")
outputTest = np.zeros((len(yTest)+1, 2), dtype='|S5')
outputTest[0,0] = "id"
outputTest[0,1] = "label"

for i in range (outputTest.shape[0]-1):
    outputTest[i+1,0] = str(i)
    outputTest[i+1,1] = str(yTest[i])

np.savetxt(sys.argv[2], outputTest, delimiter=",", fmt = "%s")

print("complete!")