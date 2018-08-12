#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def object_to_int(object_, object_list):
    xs = sorted([x for x in object_list if x is not np.nan])
    for i, x in enumerate(xs):
        if x == object_:
            return i

    return np.nan


def column_object_to_int(data, column_name):
    data[column_name] = data.apply(
        lambda row: object_to_int(row[column_name], data[column_name].unique()),
        axis=1)


def get_X_y(raw_data):
    raw_data = raw_data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    column_object_to_int(raw_data, 'Sex')
    column_object_to_int(raw_data, 'Embarked')
    X = raw_data.drop(['Survived'], axis=1)
    y = raw_data.Survived
    return X, y


def load_train_test(filename):
    raw_data = pd.read_csv(filename)
    raw_data_train, raw_data_test = train_test_split(raw_data)
    X_train, y_train = get_X_y(raw_data_train)
    X_test, y_test = get_X_y(raw_data_test)
    return Data(X_train, y_train), Data(X_test, y_test)


def train(data_train):
    return xgb.XGBClassifier().fit(data_train.X, data_train.y)


def realpath(relative_path):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, relative_path)


def main():
    data_train, data_test = load_train_test(realpath('data/train.csv'))
    model = train(data_train)
    predictions = model.predict(data_test.X)
    accuracy = accuracy_score(data_test.y, predictions)
    print('>>> accuracy: {}'.format(accuracy))
    xgb.plot_tree(model)
    plt.show()


if __name__ == "__main__":
    main()
