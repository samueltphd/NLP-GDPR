#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run the classifier,
   and print results to stdout.

   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.

   Brown CS142, Spring 2020
"""

import numpy as np
import sys
import random
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from models import LogisticRegression

ROOT_DIR_PREFIX = './data/'

DATA_FILE_NAME = 'normalized_data.csv'
# DATA_FILE_NAME = 'unnormalized_data.csv'
# DATA_FILE_NAME = 'normalized_data_nosens.csv'

CENSUS_FILE_PATH = ROOT_DIR_PREFIX + DATA_FILE_NAME

NUM_CLASSES = 3
BATCH_SIZE = 1  #tune this parameter
CONV_THRESHOLD = 1 #tune this parameter

def import_census(file_path):
    '''
        Helper function to import the census dataset

        @param:
            train_path: path to census train data + labels
            test_path: path to census test data + labels
        @return:
            X_train: training data inputs
            Y_train: training data labels
            X_test: testing data inputs
            Y_test: testing data labels
    '''
    data = np.genfromtxt(file_path, delimiter=',', skip_header=False)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X_train, Y_train, X_test, Y_test

def test_logreg():
    X_train, Y_train, X_test, Y_test = import_census(CENSUS_FILE_PATH)
    num_features = X_train.shape[1]

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###
    model = LogisticRegression(num_features, NUM_CLASSES, BATCH_SIZE, CONV_THRESHOLD)
    num_epochs = model.train(X_train_b, Y_train)
    acc = model.accuracy(X_test_b, Y_test) * 100
    print("Test Accuracy: {:.1f}%".format(acc))
    print("Number of Epochs: " + str(num_epochs))

    return acc

def main():

    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0)
    np.random.seed(0)

    test_logreg()

if __name__ == "__main__":
    main()
