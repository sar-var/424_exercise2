# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    lst = lst[0:5]
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (coef, name)
                                   for coef, name in lst)

train = pd.read_csv('train.csv')
train = train[np.isfinite(train['gpa'])]
trainSet = train['challengeID'].values - 1
labels = train['gpa'].values

df = pd.read_csv('clean_data.csv')
features = df.iloc[trainSet]

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.0000)
sfm.fit(features, labels)
n_features = sfm.transform(features).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 10:
    sfm.threshold += 0.1
    X_transform = sfm.transform(features)
    n_features = X_transform.shape[1]

import pdb; pdb.set_trace()