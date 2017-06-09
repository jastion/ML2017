import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dot, Add
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils import np_utils
import keras.backend as K
from moviesClass import data_process

epochs = 300
batch_size = 64#128
validation_split = 0.01

if __name__ == "__main__":
    moviesClass = data_process()
    baseDir, expDir = moviesClass.get_path()
    store_path, history_data = moviesClass.history_data(expDir, epochs)

    ratingsData = pd.read_csv(os.path.join(sys.argv[1],'train.csv'), sep=',', engine='python')
    usersData = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    moviesData = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

    testData = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')
    testUsers = testData.UserID.astype('category')
    testMovies = testData.MovieID.astype('category')

    print("Data loading...")
    print('Training data: ', ratingsData.shape)
    print('Users data: ', usersData.shape)
    print('Movies data: ', moviesData.shape)

    moviesData['Genres'] = moviesData.Genres.str.split('|')
    usersData.Age = usersData.Age.astype('category')
    usersData.Gender = usersData.Gender.astype('category')
    usersData.Occupation = usersData.Occupation.astype('category')

    numMovies = ratingsData['MovieID'].drop_duplicates().max()#moviesData.shape[0]
    numUsers = ratingsData['UserID'].drop_duplicates().max()#usersData.shape[0]
    movieID = ratingsData.MovieID.values
    userID = ratingsData.UserID.values

    Y_data = ratingsData.Rating.values

    '''
    #normalize output
    Y_data = np.array(Y_data, dtype='float')
    y_std = np.std(Y_data)
    y_mean = np.mean(Y_data)
    Y_data = (Y_data - y_mean) / float(y_std)
    '''

    X1_train, X2_train, Y_train, X1_val, X2_val, Y_val = moviesClass.split_data(userID, movieID, Y_data, validation_split)

    latent_dim = 16 #20

    #MODEL

    '''
    userInput = Input(shape=[1])
    user_embedding = Embedding(numUsers+1, latent_dim, embeddings_initializer='random_normal')(userInput)
    user_vec = Flatten()(user_embedding)

    movieInput = Input(shape=[1])
    movie_embedding = Embedding(numMovies+1, latent_dim, embeddings_initializer='random_normal')(movieInput)
    movie_vec = Flatten()(movie_embedding)


    user_bias = Embedding(numUsers+1, 1, embeddings_initializer='zeros')(userInput)
    user_bias = Flatten()(user_bias)
    movie_bias = Embedding(numMovies+1, 1, embeddings_initializer='zeros')(movieInput)
    movie_bias = Flatten()(movie_bias)

    vec_inputs = Dot(axes=1)([user_vec, movie_vec])
    model_out = vec_inputs
    #model_out = Add()([vec_inputs, user_bias, movie_bias])
    '''

    userInput = Input(shape=[1]) 
    movieInput = Input(shape=[1])

    userVec = Embedding(numUsers, latent_dim, embeddings_initializer='random_normal')(userInput)
    userVec = Flatten()(userVec)

    itemVec = Embedding(numMovies, latent_dim, embeddings_initializer='random_normal')(movieInput)
    itemVec = Flatten()(itemVec)

    mergeVec = Concatenate()([userVec, itemVec])
    hidden = Dense(150,activation='relu')(mergeVec)
    hidden = Dense(50,activation='relu')(hidden)
    model_out = Dense(1)(hidden)


    '''
    vec_inputs = Concatenate()([user_vec, movie_vec])
    model = Dense(1024, activation='relu')(vec_inputs)
    model = Dropout(0.4)(model)
    model = BatchNormalization()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.4)(model)
    model = BatchNormalization()(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.3)(model)
    model = BatchNormalization()(model)
    model = Dense(32, activation='relu')(model)
    model = Dropout(0.4)(model)
    model = BatchNormalization()(model)
    model_out = Dense(1, activation='linear')(model)
    '''
    
    model = Model([userInput, movieInput], model_out)
    model.compile(loss='mse', optimizer='sgd', metrics=[moviesClass.root_mean_squared_error])

    earlyStop = EarlyStopping('val_root_mean_squared_error', patience=5, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath=os.path.join(store_path,'best_model.h5'), verbose=1, save_best_only=True, monitor='val_root_mean_squared_error', mode='min')

    csv_logger = CSVLogger(os.path.join(store_path,'record_dnn.log'))

    model.fit([X1_train, X2_train], Y_train, validation_data=([X1_val, X2_val], Y_val), batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, earlyStop, csv_logger])
    model.save(os.path.join(store_path,'model.h5'))
    plot_model(model,to_file=os.path.join(store_path,'model.png'))

    model = load_model(os.path.join(store_path,'best_model.h5'), custom_objects={'root_mean_squared_error': moviesClass.root_mean_squared_error})
    yTest = model.predict([testUsers, testMovies])
    
    '''
    #denormalized output
    yTest = yTest * y_std + y_mean
    yTest = np.around(yTest)
    '''
    
    testOutput = testData
    testOutput['UserID'] = yTest
    testOutput = testOutput.drop('MovieID', 1)
    testOutput.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
    