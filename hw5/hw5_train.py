import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN
from keras.layers.wrappers import TimeDistributed

from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers.advanced_activations import LeakyReLU, PReLU
import nltk
from nltk.corpus import stopwords
from glove_learning import data_process
import cPickle as pickle
import os

nltk.download('stopwords')
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 400
batch_size = 64
range_value = 0.5

################
###   Util   ###
################
def read_data(path,training):
	print ('Reading data from ',path)
	with open(path,'r') as f:
	
		tags = []
		articles = []
		tags_list = []
		
		f.readline()
		for line in f:
			if training :
				start = line.find('\"')
				end = line.find('\"',start+1)
				tag = line[start+1:end].split(' ')
				article = line[end+2:]
				
				for t in tag :
					if t not in tags_list:
						tags_list.append(t)
			   
				tags.append(tag)
			else:
				start = line.find(',')
				article = line[start+1:]
			
			articles.append(article)
			
		if training :
			assert len(tags_list) == 38,(len(tags_list))
			assert len(tags) == len(articles)
	return (tags,articles,tags_list)

def get_embedding_dict(path):
	embedding_dict = {}
	with open(path,'r') as f:
		for line in f:
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			embedding_dict[word] = coefs
	return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
	embedding_matrix = np.zeros((num_words,embedding_dim))
	for word, i in word_index.items():
		if i < num_words:
			embedding_vector = embedding_dict.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	return embedding_matrix

def to_multi_categorical(tags,tags_list): 
	tags_num = len(tags)
	tags_class = len(tags_list)
	Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
	for i in range(tags_num):
		for tag in tags[i] :
			Y_data[i][tags_list.index(tag)]=1
		assert np.sum(Y_data) > 0
	return Y_data

def split_data(X,Y,split_ratio):
	indices = np.arange(X.shape[0])  
	np.random.shuffle(indices) 
	
	X_data = X[indices]
	Y_data = Y[indices]
	
	num_validation_sample = int(split_ratio * X_data.shape[0] )
	
	X_train = X_data[num_validation_sample:]
	Y_train = Y_data[num_validation_sample:]

	X_val = X_data[:num_validation_sample]
	Y_val = Y_data[:num_validation_sample]

	return (X_train,Y_train),(X_val,Y_val)

def translate_Categorial2label(output, label):
	Y = []
	for row in output:
		find = [pos for pos,x in enumerate(row) if x==1]
		temp = []
		for i in find:
			temp.append(label[i])
		temp = [" ".join(temp)]
		Y.append(temp)
	return Y

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
	thresh = 0.4
	y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	tp = K.sum(y_true * y_pred)
	
	precision=tp/(K.sum(y_pred))
	recall=tp/(K.sum(y_true))
	return 2*((precision*recall)/(precision+recall))



#########################
###   Main function   ###
#########################
if __name__=='__main__':

	### read training and testing data
	(Y_data,X_data,tag_list) = read_data(train_path,True)
	(_, X_test,_) = read_data(test_path,False)
	all_corpus_temp = X_data + X_test
	print ('Find %d articles.' %(len(all_corpus_temp)))

	
	### tokenizer for all data
	tokenizer_temp = Tokenizer()
	tokenizer_temp.fit_on_texts(all_corpus_temp)
	word_index_temp = tokenizer_temp.word_index

	all_corpus = [w for w in word_index_temp if not w in stopwords.words("english")]

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(all_corpus)
	word_index = tokenizer.word_index

	### convert word sequences to index sequence
	print ('Convert to index sequences.')
	train_sequences = tokenizer.texts_to_sequences(X_data)
	test_sequences = tokenizer.texts_to_sequences(X_test)

	### padding to equal length
	print ('Padding sequences.')
	train_sequences = pad_sequences(train_sequences)
	max_article_length = train_sequences.shape[1]
	test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
	
	###
	train_tag = to_multi_categorical(Y_data,tag_list) 
	
	tag_sum_analyze = np.sum(train_tag, axis=0)

	''' #prints tag list
	print ("\n\n")
	for i in range(len(tag_list)):
		print(str(tag_list[i])+":"+str(int(tag_sum_analyze[i])))
		#print("\n\n")
	'''

	### split data to training validation set
	(X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

	
	# get mebedding matrix from glove
	print ('Get embedding dict from glove.')
	embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
	print ('Found %s word vectors.' % len(embedding_dict))
	num_words = len(word_index) + 1
	print ('Create embedding matrix.')
	embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

	#

	# build model
	print ('Building model.')
	model = Sequential()
	model.add(Embedding(num_words,
						embedding_dim,
						weights=[embedding_matrix],
						input_length=max_article_length,
						trainable=False))
	'''
	model.add(Conv1D(filters=32,kernel_size=3,padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.4))
	model.add(Conv1D(filters=64,kernel_size=3,padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.4))
	'''
	
	#model.add(Conv1D(filters=128,kernel_size=5,padding='same', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	#model.add(Dropout(0.4))
	
	#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	#model.add(Conv1D(64,5,activation='relu'))
	#model.add(Dropout(0.2))
	model.add(GRU(128,activation='tanh',dropout=0.4))
	#model.add(SimpleRNN(100,return_sequences=True))
	#model.add(TimeDistributed(Dense(38, activation='sigmoid')))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(38,activation='sigmoid'))
	model.summary()

	lr = 0.001
	adam = Adam(lr=lr,decay=1e-6,clipvalue=0.5)
	rms = RMSprop(lr = lr)
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=[f1_score])
   
	earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
	name = 'best2.hdf5' 
	checkpoint = ModelCheckpoint(filepath=name,
								 verbose=1,
								 save_best_only=True,
								 save_weights_only=False,
								 monitor='val_f1_score',
								 mode='max')
	csv_logger = CSVLogger('training_sigmoid.log')
	reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.2,patience=5, min_lr=0.00001)
	
	
	hist = model.fit(X_train, Y_train, 
					 validation_data=(X_val, Y_val),
					 epochs=nb_epoch, 
					 batch_size=batch_size,
					 callbacks=[csv_logger,checkpoint])
	
	model.load_weights('best.hdf5')
   
	Y_pred = model.predict(test_sequences)

	linfnorm = np.linalg.norm(Y_pred, axis=1, ord=np.inf)
	preds = Y_pred.astype(np.float) / linfnorm[:, None]

	preds[preds >= range_value] = 1
	preds[preds < range_value] = 0

	original_y = translate_Categorial2label(preds, tag_list)

	output_file = []

	output_file.append('"id","tags"')

	for i in range (len(original_y)):
		temp = '"'+str(i)+'"' + ',' + '"' + str(" ".join(original_y[i])) + '"'
		output_file.append(temp)

		with open(sys.argv[3],'w') as f:
			for data in output_file:
				f.write('{}\n'.format(data))



	
