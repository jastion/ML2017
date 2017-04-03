import numpy as np
def sort_ranges(inputCsv):
	arrayAge = inputCsv[:,0]
	arrayHours = inputCsv[:,5]
	arrayFn = inputCsv[:,1]

	#ageRange = np.array([[0,25],[26,45],[46,65],[66,1000]])
	#payRange = np.array([[0,25],[26,40],[41,60],[60,1000]])
	#Age Range: 0-25, 26-45, 46-65, +66
	#Pay Range: 0-25, 26-40, 41-60, 60+
	# each cut into levels None (0), Low (0 < median of the values greater zero < max) and High (>=max).

	for i in range(inputCsv.shape[0]):
		age = arrayAge[i]
		pay = arrayHours[i]
		if age >= 0 and age <= 20:
			arrayAge[i] = 0
		elif age>=21 and age <= 45:
			arrayAge[i] = 31
		elif age>=46 and age <= 65:
			arrayAge[i] = 30
		elif age>65:
			arrayAge[i] = 1
		'''
		if age >= 0 and age <= 18:
			arrayAge[i] = 0
		elif age>=19 and age <= 25:
			arrayAge[i] = 50
		elif age>=26 and age <= 45:
			arrayAge[i] = 11
		elif age>=46 and age <= 65:
			arrayAge[i] = 10
		elif age>65:
			arrayAge[i] = 1
		'''
		if pay >= 0 and pay <= 25:
			arrayHours[i] = 0
		elif pay>=26and pay <= 40:
			arrayHours[i] = 20
		elif pay>=41 and pay <= 60:
			arrayHours[i] = 2
		elif pay>61:
			arrayHours[i] = 1 



	inputCsv[:,0] = arrayAge
	inputCsv[:,5] = arrayHours

	return inputCsv

def sort_capital(inputCsv):
	arrayCapG = inputCsv[:,3]
	arrayCapL = inputCsv[:,4]

	tmpA = arrayCapG
	tmpB = arrayCapL

	idxA = np.where(tmpA[:] != 0)
	tmpA = tmpA[idxA]
	idxB = np.where(tmpB[:] != 0)
	tmpB = tmpB[idxB]

	medianGain = np.median(tmpA)
	medianLoss = np.median(tmpB)

	for i in range(inputCsv.shape[0]): 	
		gain = arrayCapG[i]
		loss = arrayCapL[i]
		if gain <= 0:
			arrayCapG[i] = 0
		elif gain < medianGain and gain > 0:
			arrayCapG[i] = 5
		elif gain >= medianGain:
			arrayCapG[i] = 70

		if loss <= 0:
			arrayCapL[i] = 70
		elif loss < medianLoss and loss > 0:
			arrayCapL[i] = 5
		elif loss >= medianLoss:
			arrayCapL[i] = 0

	inputCsv[:,3] = arrayCapG
	inputCsv[:,4] = arrayCapL
	return inputCsv

def sort_data(inputCsv,operatingType):
	#0 age
	#1 fnlwgt
	#2 sex
	#3 capital_gain
	#4 capital_loss
	#5 hours
	#6-14 Employer 			9
	#15-30 Education 		16
	#31-37 Marital 			6
	#38-52 Occupation 		16
	#53-58 Family Role  	5
	#59-63 Race 			5
	#64-105 Origin of Countries 43 

	#output
	#1 age
	#2 fnlwgt
	#3 sex
	#4 capital_gain
	#5 capital_loss
	#6-12 Employer 			8
	#13-20 Education 		15
	#21-24 Marital 			6
	#25-33 Occupation 		14
	#34-39 Family Role  	5
	#40-44 Race 			4
	#45-55 Origin of Countries 42

	arrayAge = inputCsv[:,0]
	arrayWgt = inputCsv[:,1]
	arraySex = inputCsv[:,2]
	arrayCapitalG = inputCsv[:,3]
	arrayCapitalL = inputCsv[:,4]
	arrayHours = inputCsv[:,5]
	arrayEmployer = inputCsv[:,6:15] 
	arrayEducation = inputCsv[:,15:31] 
	arrayMarital = inputCsv[:,31:38]
	arrayOccupation = inputCsv[:,38:53]
	arrayFamRole = inputCsv[:,53:59]
	arrayRace = inputCsv[:,59:64]
	arrayCountry = inputCsv[:,64:]

	if (operatingType==0):
		#14, 52, 105
		idxRow1 = np.where(inputCsv[:,14] == 1)[0]
		idxRow2 = np.where(inputCsv[:,52] == 1)[0]
		idxRow3 = np.where(inputCsv[:,105] ==1)[0]
		idxRow = np.array([])
		idxRow = np.append(idxRow,idxRow1)
		idxRow = np.append(idxRow,idxRow2)
		idxRow = np.append(idxRow,idxRow3)
		idxRow = np.unique(idxRow)
	else:
		idxRow = np.array([])
	'''
	idxEmployer = np.array([[6,7,12],[8,13],[9],[10,11]])
	idxEducation = np.array([[15,16,17,18,19,20,21,28],[22,23,26,30],[24,25,27],\
													[29]])
	idxMarital = np.array([[31,36],[32,33,34,37],[35]])
	#idxMarital = np.array([[31,36],[32,33],[34,35],[37]])
	idxOccupation = np.array([[38,49,50],[39],[40,42,43,44,51],[41],[45,46],[47],\
													[48]])
	idxFamRole = np.array([[53],[54],[55],[56],[57],[58]])
	idxRace = np.array([[59],[60],[61],[62],[63]])
	idxCountry = np.array([[64,88,93,100,103],[65,72,82,84,97],[66,80,99],\
				[67,70,71,92],[68,83,87],[69,76,77,79,86,89,90,91,96,101],\
				[73,74,78,85],[75,81,94,95,98,104],[102]])
	'''
	idxEmployer = np.array([[6],[7,12],[8,13],[9],[10,11]])
	idxEducation = np.array([[15,16,17,18,19,20,21,28],[22,23],[24],[25],\
													[26,30],[27],[29]])
	idxMarital = np.array([[31,36],[32,33,34,37],[34,35]])
	#idxMarital = np.array([[31,36],[32,33],[34,35],[37]])
	idxOccupation = np.array([[38],[39],[40,42,43,44,51],[41],[45,46],[47],\
													[48,50],[49]])
	idxFamRole = np.array([[53],[54],[55],[56],[57],[58]])
	idxRace = np.array([[59],[60],[61],[62],[63]])
	idxCountry = np.array([[64,88,93,100,103],[65,72,82,84,97],[66,80,99],\
				[67,70,71,92],[68,83,87],[69,76,77,79,86,89,90,91,96,101],\
				[73,74,78,85],[75,81,94,95,98,104],[102]])
	

	finalArray = arrayAge
	#finalArray = arrayWgt
	finalArray = np.vstack((arrayAge,arrayWgt))
	finalArray = np.vstack((finalArray,arraySex.reshape(1,arraySex.shape[0])))
	finalArray = np.vstack((finalArray,arrayCapitalG.reshape(1,arrayCapitalG.shape[0])))
	finalArray = np.vstack((finalArray,arrayCapitalL.reshape(1,arrayCapitalL.shape[0])))
	finalArray = np.vstack((finalArray,arrayHours.reshape(1,arrayHours.shape[0])))

	#idxArray = np.array([[idxEmployer],[idxEducation],[idxMarital], [idxOccupation], [idxFamRole],[idxRace],[idxCountry]])
	idxArray = np.array([[idxEmployer],[idxEducation],[idxMarital], [idxOccupation], [idxFamRole],[idxCountry]])
	#idxArray = np.array([[idxFamRole],[idxRace],[idxCountry]])
	#idxArray = np.array([[idxEmployer],[idxMarital],[idxEducation],[idxOccupation]])

	#finalArray = finalArray.reshape(1,inputCsv.shape[0])
	for index in range(len(idxArray)):
		currSelection = idxArray[index]
		#Selects array from idxArray
		for idx in range(len(currSelection)):
			#Selects Array containing array of  column groups to combine
			for i in range(len(currSelection[idx])):
				currColumn = currSelection[idx]
				#iterate through values in array
				tmpHolder = np.zeros((1, inputCsv.shape[0]))
				for k in range(len(currColumn[i])):

					tmpHolder += inputCsv[:,currColumn[i][k]]
				tmpHolder = np.clip(tmpHolder,0,1)
				finalArray = np.vstack((finalArray,tmpHolder))
	finalArray = finalArray.T
	finalArray = np.delete(finalArray,idxRow,axis=0)

	return finalArray, idxRow

def cut_data(inputCsv):


	age = np.array([0])
	fnlwgt = np.array([1])
	sex = np.array([2])
	capital_gain = np.array([3])
	capital_loss = np.array([4])
	hours = np.array([5])
	Employer = np.arange(6,15)
	Education = np.arange(15,31)
	Marital = np.arange(31,38)
	Occupation = np.arange(38,53)
	Family_Role = np.arange(53,59)
	Race = np.arange(59,64)
	Origin_of_Countries = np.arange(64,106)

	a = np.array([age,fnlwgt,sex,capital_gain,capital_loss,Employer,Education,Marital,Occupation,Family_Role,Race,Origin_of_Countries])
	a = np.array([fnlwgt,capital_gain,Education,Marital,Occupation,Family_Role])

	idxArray = np.concatenate(a)
	outputCsv = inputCsv[:,idxArray]

	return outputCsv
	
def remove_unknown(inputCsv):
	print(inputCsv.shape)
	idxRow1 = np.where(inputCsv[:,14] == 1)[0]
	idxRow2 = np.where(inputCsv[:,52] == 1)[0]
	idxRow3 = np.where(inputCsv[:,105] == 1)[0]

	idxRow = np.array([])
	idxRow = np.append(idxRow,idxRow1)
	idxRow = np.append(idxRow,idxRow2)
	idxRow = np.append(idxRow,idxRow3)
	idxRow = np.unique(idxRow)
	outputCsv = np.delete(inputCsv,idxRow,axis=0)
	outputCsv = np.delete(outputCsv,[14,52,105],axis=1)
	print(outputCsv.shape)
	return outputCsv, idxRow

def balance_data(inputCsv):

	idxRowValA = np.where(inputCsv[:,-1] == 0)[0]
	idxRowValB = np.where(inputCsv[:,-1] == 1)[0]

	arrayDataA = inputCsv[idxRowValA,:]
	arrayDataB = inputCsv[idxRowValB,:]

	arrayDataA=arrayDataA[0:arrayDataB.shape[0],:]


	#arrayDataC = np.vstack((arrayDataB,arrayDataB))
	#arrayDataB = np.vstack((arrayDataB,arrayDataB))

	outputCsv = np.vstack((arrayDataA,arrayDataB))
	np.random.shuffle(outputCsv)


	return outputCsv


