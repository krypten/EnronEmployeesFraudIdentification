#!/usr/bin/python

import sys
import pickle
sys.path.append("../lib/")

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

with open("../data/final_project_dataset.pkl", "r") as data_file:
    my_dataset = pickle.load(data_file)

from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

print (len(features_list))
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_list = features_list[1:]
X, y = make_classification(n_samples=len(features), n_features=len(features_list))

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear",C=100)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Total number of features : %d" % len(features_list))
print("Optimal number of features : %d" % rfecv.n_features_)
print("List of features : {}".format(features_list))
print("Ranking of features : {}".format(sorted(zip(features_list, rfecv.ranking_), key=lambda x: x[1])))
print len(rfecv.ranking_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

