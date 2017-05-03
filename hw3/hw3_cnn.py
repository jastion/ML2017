import sys
import cnn as CNN

#https://docs.google.com/presentation/d/1QFK4-inv2QJ9UhuiUtespP4nC5ZqfBjd_jP2O41fpTc/edit#slide=id.p
#https://inclass.kaggle.com/c/ml2017-hw3
batch_size = 64

validation =0.2
nb_epoch = 10
alpha = 0.001

#Conv2D layerNum,typeAct,poolNum,dropNum):
#Dense layerNum,dropNum

#Training File, #Testing File, #Augment Data Flag, #ZMUV Flag
network = CNN.ConvNet(sys.argv[1],sys.argv[2],0,0)
typeAct = 0 #0 = relu 1 = LeakyRelu
#Conv2D layerNum,typeAct,poolNum,dropNum):
#alpha = leakyReLu
network.model_generate()

network.init_conv2D_layer(64,typeAct,2,0.2,alpha)
network.add_conv2D_layer(128,typeAct,2,0.3,alpha)
network.add_conv2D_layer(250,typeAct,2,0.3,alpha)
network.add_conv2D_layer(500,typeAct,2,0.5,alpha)

network.flatten_model()
#layerNum, dropout
network.add_dense_layer(64,0.2)
network.add_dense_layer(128,0.2)
network.add_dense_layer(256,0.5)

network.add_softmax_layer(7)
#batch_size, epoch, validation size, lr, method (regular, batch, SGD)
network.run_model(batch_size,nb_epoch,0.2,0.01,1) #0.001

network.predict_test_data(batch_size)

count = 5


'''
while (count > 0):
	network.model_generate()
	network.init_conv2D_layer(64,typeAct,2,0,alpha)
	network.add_conv2D_layer(128,typeAct,2,0,alpha)
	network.add_conv2D_layer(250,typeAct,2,0,alpha)
	network.add_conv2D_layer(500,typeAct,2,0,alpha)
	network.flatten_model()
	#layerNum, dropout
	network.add_dense_layer(64,0.2)
	network.add_dense_layer(128,0.2)
	network.add_dense_layer(256,0.5)
	network.add_softmax_layer(7)
	#batch_size, epoch, validation size
	network.run_model(batch_size,nb_epoch,0.2)

	network.predict_test_data()
'''