import os
import sys
import collections
import nltk
import numpy as np
from nltk.corpus import stopwords
from numpy.linalg import norm
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN
from keras.models import load_model
from keras.layers.embeddings import Embedding
from glove_learning import data_process


maxlen_pad_sequences = 190
epoch = 500
batch_size = 64
validation = 0.1
gap_probability = 0.42


if __name__ == "__main__":
    data_process = data_process(gap_probability)

    (_, X_test,_) = data_process.read_data(sys.argv[1],False)

    tag_list = data_process.load_pickle('label.pickle')
    tokenizer = data_process.load_pickle('tokenizer.pickle')
    #num_words = data_process.load_pickle('num_words.pickle')
    #embedding_dim = data_process.load_pickle('embedding_dim.pickle')
    #embedding_matrix = data_process.load_pickle('embedding_matrix.pickle')
    #max_article_length = data_process.load_pickle('max_article_length.pickle')
    #test_sequences = data_process.load_pickle('test_sequences.pickle')

    '''
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))

    
    model.add(GRU(128,activation='tanh',dropout=0.4))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38,activation='sigmoid'))
    model.load_weights('best.hdf5')
    '''
    model = load_model('best.hdf5',  custom_objects={'f1_score':data_process.f1_score})
    
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    test_sequences = pad_sequences(test_sequences, maxlen=maxlen_pad_sequences)

    #predict 
    test_Y = model.predict(test_sequences)
    
    #normalize probability
    linfnorm = norm(test_Y, axis=1, ord=np.inf)
    test_Y = test_Y.astype(np.float) / linfnorm[:, None]

    test_Y[test_Y>=gap_probability] = 1
    test_Y[test_Y<gap_probability] = 0

    output_file = []
    output_file.append('"id","tags"')

    for i,labels in enumerate(test_Y):
        labels = [tag_list[x] for x,value in enumerate(labels) if value==1 ]
        if(labels==[]):
            labels=["FICTION"]
        temp = '"'+str(i)+'"' + ',' + '"' + str(" ".join(labels)) + '"'
        output_file.append(temp)

    with open(sys.argv[2],'w') as f:
        for data in output_file:
            f.write('{}\n'.format(data))
