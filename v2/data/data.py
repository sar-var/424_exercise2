from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import dill


class FullDataImporter(object):
    def __init__(self):
        # import data
        features = pd.read_csv("../imputation/imputed-large.csv")

        # Create a boolean mask for likely categorical features
        msk_int = (features.dtypes == "int64").tolist()
        msk_cat = [x < 6 for x in features.apply(lambda x: len(set(x)), axis=0,
                                                 raw=True).tolist()]  # gpa.apply(lambda x: len(set(x)), axis = 0, raw = True)
        msk = (msk_int and msk_cat)

        # Convert factors into factor codes
        factors = features.loc[:, msk]
        factors = factors.apply(lambda x: x.astype('category'))
        factors_extended = pd.get_dummies(factors)

        # Assume all other data is continuous
        continuous = features.loc[:, [not x for x in msk]]
        challengeIds = pd.DataFrame(continuous, columns=["challengeID"])
        continuous = continuous.drop("challengeID", axis=1)
        ccolnames = continuous.columns

        # Scale continuous data -- better for some classifiers such as linear models/SVM
        scaler = StandardScaler()
        continuous = pd.DataFrame(scaler.fit_transform(continuous))
        continuous.columns = ccolnames

        self.features = pd.concat([challengeIds, factors_extended, continuous], axis=1)
        self.outcomes = pd.read_csv("../data/train.csv")

        self.name = "full"

    def all_features(self, feature=None):
        return self.features.drop("challengeID", axis=1)

    def x_y_for_feature(self, feature):
        # select only our feature of interest
        single_outcome = pd.DataFrame(self.outcomes, columns=["challengeID", feature])
        # remove all NA
        train = single_outcome[np.isfinite(single_outcome[feature])]
        # y values
        y = train[feature]
        # X values for all relevant features
        X = self.features.loc[self.features["challengeID"].isin(list(train["challengeID"]))].drop("challengeID", axis=1)
        return X, y


class SelectionDataImporter(FullDataImporter):
    def __init__(self):
        super(SelectionDataImporter, self).__init__()
        self.mask = None  # set by subclasses

    def x_y_for_feature(self, feature):
        # select only our feature of interest
        single_outcome = pd.DataFrame(self.outcomes, columns=["challengeID", feature])
        # remove all NA
        train = single_outcome[np.isfinite(single_outcome[feature])]
        # y values
        y = train[feature]

        # X values for all relevant features
        X = self.features.loc[self.features["challengeID"].isin(list(train["challengeID"]))].drop("challengeID", axis=1)
        X = X.loc[:, self.mask[feature]]
        return X, y

    def all_features(self, feature=None):
        if feature is None:
            raise ValueError("You need to specify a feature name!")
        return self.features.drop("challengeID", axis=1).loc[:, self.mask[feature]]


class LassoDataImporter(SelectionDataImporter):
    def __init__(self):
        super(LassoDataImporter, self).__init__()
        self.mask = {
            "gpa": dill.load(open("../featuremasks/model-lasso-gpa.p")),
        }
        self.name = "lasso"


