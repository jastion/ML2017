import numpy as np
import string
import sys
import keras.backend as K
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU,Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer,normalize
from sklearn.decomposition import PCA, TruncatedSVD

K.clear_session()
#nltk.download('stopwords')
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 300
batch_size = 512
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
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus_temp = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus_temp)))

    

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=2, max_df  = 0.7)
    all_corpus_temp = vectorizer.fit_transform(all_corpus_temp)
    # normalize
    normalizer = Normalizer(norm='l2',copy=True)
    all_corpus_temp = normalize(all_corpus_temp, norm='l2')

    all_corpus_temp = all_corpus_temp.toarray()
    X_data = all_corpus_temp[0:4964,:]
    X_test = all_corpus_temp[4964:6198,:]


    print ('Padding sequences.')

    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(X_data,train_tag,split_ratio)
    
    model = Sequential()
    model.add(Dense(128,input_shape=X_train.shape[1:],activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file = output )
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    