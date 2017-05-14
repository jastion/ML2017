import sys
import Image as img
import numpy as np 
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy import linalg
import PIL
import os

class PCA:
	"""docstring for PCA"""
	def __init__(self,path,count_num,char_count_num):
		trainData = []
		idx = "A"
		count = 0
		char_count = 1
		for file in sorted(os.listdir(path)):
			if file.endswith(".bmp"):
				if (file.startswith(idx) and (count < count_num)):
					filepath = path + file
					imgArr = np.asarray(img.open(filepath))
					imgList = np.reshape(imgArr, (np.product(imgArr.shape), )).astype('int')
					trainData.append(imgList)
					count += 1
				elif ((char_count < char_count_num) and (count == count_num)):
					count = 0
					char_count += 1
					idx = chr(ord(idx)+1)
		trainData = np.asarray(trainData)
		self.trainData = trainData

	def column(self,matrix, i):
		return [row[i] for row in matrix]
	
	def find_mean(self):
		dataMean =[]
		trainData = self.trainData
		for idx in np.arange(len(trainData[0])):
			meanValue = np.mean(self.column(trainData,idx))
			dataMean.append(meanValue)
		dataMean = np.asarray(dataMean)
		self.dataMean = dataMean
		return dataMean

	def center_data(self):
		dataMean = self.dataMean
		trainData = self.trainData
		for idx in np.arange(len(self.column(trainData,0))):
			trainData[idx] = trainData[idx] - dataMean
		dataAdjust = np.asarray(trainData)
		self.dataAdjust = dataAdjust
		return dataAdjust

	def uncenter_data(self,matrix):
		mean = self.dataMean
		reconSet =[]
		for idx in np.arange(len(matrix)):
			reconSet.append(matrix[idx]+mean)
		reconSet = np.asarray(reconSet)
		self.dataOrig = reconSet
		return reconSet

	def SVD(self,matrix):
		[u,s,v] = svd(matrix)
		self.u = u
		self.s = s
		#v = vt.T
		#self.vt = vt
		self.v = v
		return u,s,v

	def find_eigenface(self,u,s,vectorNum):
		mean = self.dataMean
		eigenFace = []
		tmp = np.dot(u,s)
		tmp = tmp.T
		for idx in np.arange(vectorNum):
			eigenFace.append(tmp[idx]+mean)
		eigenFace = np.asarray(eigenFace)
		return eigenFace

	def reduce_dimension(self,topEigenNum):
		s_reduced = []
		s = self.s
		for idx in np.arange(len(s)):
			if (idx < topEigenNum ):
				s_reduced.append(s[idx])
			else:
				s_reduced.append(0)
		s_reduced = np.asarray(s_reduced)
		s_reduced = linalg.diagsvd(s_reduced, 4096, 100)
		return s_reduced

	def reconstruct_data(self,u,s,v):
		mean = self.dataMean
		reconSet =[]
		recon = np.dot(np.dot(u,s),v)
		recon_t = recon.T
		for idx in np.arange(len(recon_t)):
			reconSet.append(recon_t[idx]+mean)
		reconSet = np.asarray(reconSet)
		return reconSet


	def save_image(self,inputArr,imgNum,name):
		num = int(np.sqrt(imgNum))
		imgArr = np.asarray(inputArr).reshape(len(inputArr),64,64)
		fig = plt.figure()
		fig.canvas.set_window_title(name)
		for idx in np.arange(imgNum):
			plt.subplot(num,num,idx+1)
			plt.imshow(imgArr[idx],cmap='gray')
			plt.axis('off')

		plt.savefig(name,transparent=True, bbox_inches='tight', pad_inches=0)
		#plt.show()

	def plot_average_face(self):
		meanList = self.dataMean
		averageFace =np.asarray(meanList).reshape(64,64)
		fig = plt.figure()
		fig.canvas.set_window_title('Average Faces')
		plt.imshow(averageFace, cmap='gray')
		plt.savefig("hw4_mean_face.jpg",transparent=True, bbox_inches='tight', pad_inches=0)
		#plt.show()


	def plot_error(self):
		errorList =[]
		flag = 0
		u = self.u
		s = self.s
		v = self.v
		data = self.dataOrig
		for idx in np.arange(100):
			s_red = self.reduce_dimension(idx+1)
			eigenFace = self.find_eigenface(u,s_red,idx+1)
			recon = self.reconstruct_data(u,s_red,v)

			recon = recon.astype(np.float64)

			error = np.sqrt(((data-recon)**2).mean())/256*100
			if (flag == 0 and error < 1.0):
				print("Error less than 1%% k=%d" % (idx+1))
				flag = 1
			errorList.append(error)

		x = np.arange(100)
		y = np.ones(100,int)

		errorList = np.asarray(errorList).astype(np.float64)
		errorList = errorList
		fig = plt.figure()
		fig.canvas.set_window_title('Error vs Kth Dimensions')
		plt.plot(errorList)
		plt.plot(x,y)
		plt.title("Error vs K Dimensions")
		plt.xlabel('K Dimensions')
		plt.ylabel('Error Percentage')
		plt.savefig("hw4_p1_error_3.jpg",transparent=True, bbox_inches='tight', pad_inches=0)
		#plt.show()

		return 0