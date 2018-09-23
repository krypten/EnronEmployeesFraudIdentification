#!/usr/bin/python


def outlierCleaner(predictions, features, labels):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (feature, labeel, error).
    """

    error = [(label - prediction)**2  for label, prediction in zip(labels, predictions)]
    cleaned_data = zip(features, labels, error)

    ### your code goes here
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    cleaned_data = cleaned_data[:int(0.9 * len(cleaned_data))]
    return cleaned_data
