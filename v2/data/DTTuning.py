import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from data import FullDataImporter
from sklearn.model_selection import cross_val_score

# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
data = FullDataImporter()
X, y = data.x_y_for_feature("gpa")

# creating odd list of K for KNN
myList = list(range(0, 30))

# subsetting just the odd ones
depths = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for d in depths:
    dt = DecisionTreeRegressor(max_depth=d)
    scores = cross_val_score(dt, X, y, cv=10)
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_depth = depths[MSE.index(min(MSE))]
print "The optimal depth is %d" % optimal_depth