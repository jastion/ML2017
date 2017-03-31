#!/bin/bash

#$1: raw data (train.csv)  $2: test data (test.csv)  
#$3: provided train feature (X_train)  $4: provided train label (Y_train)
#$5: provided test feature (X_test)     $6: prediction.csv

#./hw2_generative.sh train.csv test.csv X_train Y_train X_test prediction.csv
python2.7 hw2_gen.py $1 $2 $3 $4 $5 $6