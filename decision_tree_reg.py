# Code from sklearn documentation

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def decision_tree_reg():

    X_data = pd.read_csv('features_sv2.csv')
    y_data = pd.read_csv('labels_sv2.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)

    # Predict
    pred_1 = regr_1.predict(X_test)
    pred_2 = regr_2.predict(X_test)

    acc_1 = regr_1.score(X_test, y_test)
    acc_2 = regr_2.score(X_test, y_test)

    print(acc_1)
    print(acc_2)

    # Plot the results
    '''plt.figure()
    plt.scatter(X_train, y_train, s=20, edgecolor="black",
                c="darkorange", label="data")
    plt.plot(X_test, pred_1, color="cornflowerblue",
            label="max_depth=2", linewidth=2)
    plt.plot(X_test, pred_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()'''

if __name__ == '__main__':
    decision_tree_reg()