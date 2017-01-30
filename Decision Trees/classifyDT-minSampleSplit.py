from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## DECISION TREE #################################

### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively

def classify(features, labels, samples):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = samples)
    clf = clf.fit(features, labels)
    return clf

clf2 = classify(features_train, labels_train, 2)
acc_min_samples_split_2 = clf2.score(features_test,labels_test)

clf50 = classify(features_train, labels_train, 50)
acc_min_samples_split_50 = clf50.score(features_test,labels_test)

def submitAccuracies():
    return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
print submitAccuracies()