#!/usr/bin/python
"""
======================
Precision-Recall Curve
======================

In information retrieval, precision is a measure of result relevancy, while
recall is a measure of how many truly relevant results are returned. A high
area under the curve represents both high recall and high precision, where high
precision relates to a low false positive rate, and high recall relates to a
low false negative rate. High scores for both show that the classifier is
returning accurate results (high precision), as well as returning a majority of
all positive results (high recall).

A system with high recall but low precision returns many results, but most of
its predicted labels are incorrect when compared to the training labels. A
system with high precision but low recall is just the opposite, returning very
few results, but most of its predicted labels are correct when compared to the
training labels. An ideal system with high precision and high recall will
return many results, with all results labeled correctly.

Precision (:math:`P`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false positives
(:math:`F_p`).

:math:`P = \\frac{T_p}{T_p+F_p}`

Recall (:math:`R`) is defined as the number of true positives (:math:`T_p`)
over the number of true positives plus the number of false negatives
(:math:`F_n`).

:math:`R = \\frac{T_p}{T_p + F_n}`

These quantities are also related to the (:math:`F_1`) score, which is defined
as the harmonic mean of precision and recall.

:math:`F1 = 2\\frac{P \\times R}{P+R}`

"""

import matplotlib.pyplot as plt
from itertools import cycle

def plotOutliers(x, y, labels):
    plt.clf()
    for i in range(len(x)):
        color = 'darkorange' if (labels[i] == 0) else 'navy'
        plt.scatter(x[i], y[i], lw=2, color=color)
    # plt.scatter(x, y)
    plt.title('Outlier Detection')
    plt.show()

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

def plotPrecisionRecallCurve(precision, recall):
    plt.clf()
    for i, color in zip(range(len(precision)), colors):
        plt.plot(recall[i], precision[i], lw=lw, color=color,
         label='Precision-Recall curve for feature {}'.format(i))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

