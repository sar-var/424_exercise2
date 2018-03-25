import argparse
import sys
import time
import pandas as pd
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold, KFold
import dill

from data import FullDataImporter

data = FullDataImporter()
STEPS = 5
FOLDS = 10
VERBOSE = 1
JOBS = 2

def parallel_feature_selection(func_extract_regression):
    processes = []
    print('here 0')
    X, y = data.x_y_for_feature("gpa")
    extract_regression_features_model("gpa", X, y)


def feature_selection_model():
    parallel_feature_selection(extract_regression_features_model)


def extract_regression_features_model(feature, X, y):
    clf = SelectFromModel(LassoCV(max_iter=100, cv=FOLDS, n_jobs=JOBS, verbose=VERBOSE))
    clf.fit(X, y)
    print("Selected " + str(sum(clf.get_support())) + " features for " + feature + " using LassoCV.")
    dill.dump(clf.get_support(), open("../data/model-lasso-" + feature + ".p", "wb"))



if __name__ == '__main__':
    print("Starting selection with Lasso.")
        feature_selection_model()

