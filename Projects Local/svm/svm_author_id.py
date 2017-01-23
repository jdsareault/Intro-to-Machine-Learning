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

### My work starts here
# Import SVM Package from SciKit Learn
from sklearn import svm

    # Create classifier
c = 10000
clf = svm.SVC(kernel = 'rbf', C = c)

    # Decimate data by a factor of 100 to reduce training time
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

    # Train classifier, time training process
t0 = time()
clf.fit(features_train, labels_train)
print "Training Time:", round(time()-t0,3), "s"

    # Create predictions list. Print desired results
pred = clf.predict(features_test)
#print "Item 10 is", pred[10]
#print "Item 26 is", pred[26]
#print "Item 50 is", pred[50]
print "Chris sent", pred.sum(), "emails"

    #Evaluate accuracy of classifier based on the test data. Print result
accuracy = clf.score(features_test, labels_test)
print("The accuracy is " + str(accuracy) + " using C value " + str(c))
