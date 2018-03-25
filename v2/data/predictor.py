import pandas as pd
import os
from sklearn.model_selection import GridSearchCV


class Predictor(object):
    def __init__(self, regressor, data, ext):
        """
        :type regressor: RegressorMixin
        :type classifier: ClassifierMixin
        :type data: DataImporter
        :type ext: str
        """
        self.regressor = regressor
        self.ext = ext
        self.data = data

        self.regression_features = ["gpa"]
        self.prediction = pd.DataFrame([x for x in range(1, 4243)])

    def predict_all(self):
        for f in self.regression_features:
            print("Predicting feature " + f)
            prediction = self._predict_regression(f)
            prediction = pd.DataFrame(prediction)
            self.prediction = pd.concat([self.prediction, prediction], axis=1)

        self.prediction.columns = ["challengeID"] + self.regression_features 

        directory = "../predictions/" + self.data.name + "-" + self.ext
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.prediction.to_csv(directory + "/prediction.csv", index=False)

    def _predict_regression(self, feature):
        X_train, y_train = self.data.x_y_for_feature(feature)
        self.regressor.fit(X_train, y_train)

        if isinstance(self.regressor, GridSearchCV):
            print("Optimal parameters for feature " + feature + " are:")
            print(self.regressor.best_params_)

        predictions = self.regressor.predict(self.data.all_features(feature))
        return predictions

