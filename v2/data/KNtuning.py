import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from data import FullDataImporter
from sklearn.model_selection import cross_val_score

# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
data = FullDataImporter()
X, y = data.x_y_for_feature("gpa")

# creating odd list of K for KNN
myList = list(range(49,60))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10)
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k