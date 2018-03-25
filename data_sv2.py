import numpy as np
import pandas as pd

def data_aana():

    train = pd.read_csv('train.csv')
    train = train[np.isfinite(train['gpa'])]
    #train.to_csv('test1.csv')

    #trainSet = train['challengeID'].values - 1 # previous version

    trainSet = train['challengeID'].values
    print(trainSet)

    # drop all columns but gpa
    train = train.drop(columns=["challengeID","grit","materialHardship","eviction","layoff","jobTraining"])
    # drop top row with labels
    train = train.drop(train.columns[0], axis=1)
    train = train.drop([0])

    df = pd.read_csv('clean_data.csv')
    features = df.iloc[trainSet]
    features = features.drop(features.columns[0, 1], axis=1)
    features = features.drop([0])

    features.to_csv('features_aana.csv')
    train.to_csv('labels_aana.csv')


if __name__ == '__main__':
    data_aana()
