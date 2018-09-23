#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../lib/")

# from outlier import outlierCleaner
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import IsolationForest
from plotting import plotOutliers
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from plotting import plotOutliers
# from dumper import dump_classifier_and_data

# Data Exploration
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# print(data_dict)

my_dataset = data_dict
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#########################
# Outlier Investigation #
#########################

features_outlier = SelectKBest(f_classif, k=3).fit_transform(PCA().fit_transform(features, labels), labels)
plotOutliers(np.array(features_outlier)[:,0], np.array(features_outlier)[:,2], labels)
clf = IsolationForest(max_samples=len(features_outlier), contamination=0.25, random_state=42).fit(features_outlier)

scores_pred = clf.decision_function(features_outlier)
cleaned_data = zip(labels, features, scores_pred)
cleaned_data = sorted(cleaned_data,key=lambda x:x[2])
# print cleaned_data[int(len(features)*0.1)]
cleaned_data = cleaned_data[int(len(features)*0.05):]

labels, features, _ = zip(*cleaned_data)
# features = features_outlier;
#print 'Labels {}'.format(np.array(features)[:,1])

'''
from sklearn import linear_model
reg = linear_model.LinearRegression().fit(features_train, labels_train)
predictions = reg.predict(features_train)
cleaned_data = outlierCleaner(predictions, features_train, labels_train)
if len(cleaned_data) > 0:
    features, labels, errors = zip(*cleaned_data)
    features = numpy.reshape( numpy.array(features), (len(features), 1))
    labels = numpy.reshape( numpy.array(labels), (len(labels), 1))

    reg.fit(features, labels)
    print (reg.coef_)
    import matplotlib.pyplot as plt
    plt.plot(ages, reg.predict(ages), color="blue")
    plt.show()
'''
#'''
# my_dataset = data_dict

#################################
# Feature Selection/Engineering #
#################################

# featuresSelect = SelectPercentile(f_classif, percentile=40)
# features = featuresSelect.fit_transform(features, labels)
# print "Number of features after selection {}".format(len(features[0]))
# labels, features = targetFeatureSplit(data)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
for train_data, test_data in skf.split(features, labels):
    #from sklearn.cross_validation import train_test_split
    #features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4, random_state=42)
    import 
    features_train
    

    ############################
    # Classification Algorithm #
    ############################
    pca = PCA(n_components=min(len(features_train), len(features_train[0])))
    features_train = pca.fit_transform(features_train)
    print "PCA Explained Variance Ratio : {}".format(pca.explained_variance_ratio_) 

    param_grid = {
        "n_estimators": [50, 100, 200, 1000]
    }

    clf = AdaBoostClassifier() # GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
    clf = clf.fit(features_train, labels_train);

    predictions = clf.predict(features_test);

    #############################
    # Validation and Evaluation #
    #############################
    accuracy = accuracy_score(labels_test, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels_test, predictions)

    print ("Accuracy Score : {}".format(accuracy))
    print ("Precision : {}".format(precision))
    print ("Recall : {}".format(recall))
    print ("F1 Score : {}".format(f1_score))
#'''