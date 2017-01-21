#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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

### My work starts here
#Import Gaussian Naive Bayes package from SciKit Learn
from sklearn.naive_bayes import GaussianNB

#Create & train a classifier. Measur the training time
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "Training Time:", round(time()-t0,3), "s"
#Predict the outcome of the test data based on the classifier
t1 = time()
predict = clf.predict(features_test)
print "Prediction Time:", round(time()-t1,3), "s"

#Evaluate accuracy of classifier based on the test data. Print result
accuracy = clf.score(features_test, labels_test)
print("The accuracy is " + str(accuracy))
