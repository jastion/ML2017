import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Concatenate, Dot, Add
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from movies_class import data_process

if __name__ == "__main__":
    moviesClass = data_process()

    ratingsData = pd.read_csv(os.path.join(sys.argv[1],'train.csv'), sep=',', engine='python')
    usersData = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    moviesData = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

    testData = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')
    testUsers = testData.UserID.astype('category')
    testMovies = testData.MovieID.astype('category')

    '''
    numUserGender = {"M":0, "F":1}
    usersData.Gender = usersData.Gender.replace(numUserGender)
    usersData.Gender = usersData.Gender.astype('int')
    

    testUsers = testData.UserID.values

    testUserDetailsInput = []

    for i in range(len(testUsers)):
        testUserDetailsInput.append(usersData[usersData.UserID == testUsers[i]].values[0][0:4])

    testUsers = np.asarray(testUserDetailsInput, dtype='int')
    '''

    model = load_model('best_model_first.h5', custom_objects={'root_mean_squared_error': moviesClass.root_mean_squared_error})
    yTest = model.predict([testUsers, testMovies])
    
    testOutput = testData
    testOutput['UserID'] = yTest
    testOutput = testOutput.drop('MovieID', 1)
    testOutput.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
    