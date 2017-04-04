#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

c_variations = [10, 100, 1000, 10000]
# Added in from Quiz: A Smaller Training Set and used for optimizing C
# features_train = features_train[:len(features_train) / 100]
# labels_train = labels_train[:len(labels_train) / 100]

for C in c_variations:
    kernel = 'rbf' #'linear'
    # Changed to a 'rbf' kernel from linear in Quiz: Deploy An RBF Kernel
    svm = SVC(kernel=kernel, C=C)

    print "\nC Variations: kernel={}, C={}".format(kernel, C)
    t0 = time()
    svm.fit(features_train, labels_train)
    print "Training time: {} s".format(round(time() - t0, 3))

    t0 = time()
    predictions = svm.predict(features_test)
    print "Prediction Time: {} s".format(round(time() - t0, 3))

    accuracy = svm.score(features_test, labels_test)
    print "Accuracy was {}%".format(100 * round(accuracy, 3))
#########################################################

"""
Results:

no. of Chris training emails: 7936
no. of Sara training emails: 7884
Training time: 127.873 s
Prediction Time: 13.226 s
Accuracy was 98.4%


After truncating the training set:

no. of Chris training emails: 7936
no. of Sara training emails: 7884
Training time: 0.069 s
Prediction Time: 0.746 s
Accuracy was 88.5%
"""

