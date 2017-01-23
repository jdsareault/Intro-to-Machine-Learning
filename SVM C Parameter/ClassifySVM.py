def classify(features_train, labels_train, c):
    ### import the sklearn module for GaussianNB
    from sklearn import svm
    ### create classifier
    clf = svm.SVC(C = c, kernel = 'rbf', gamma = 1.0)

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### return the fit classifier
    return clf
