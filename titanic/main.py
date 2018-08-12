#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


class CSV:
    def __init__(self, filename):
        self.raw_data = pd.read_csv(realpath(filename))

    def train_test_split(self):
        raw_data_train, raw_data_test = train_test_split(self.raw_data)
        return self.__parse_data(raw_data_train), self.__parse_data(
            raw_data_test)

    def __parse_data(self, raw_data):
        data = raw_data.drop(
            ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        self.__column_object_to_int(data, 'Sex')
        self.__column_object_to_int(data, 'Embarked')
        X = data.drop(['Survived'], axis=1, errors='ignore')
        y = data.get('Survived')
        return Data(X, y)

    def __column_object_to_int(self, data, column_name):
        data[column_name] = data.apply(
            lambda row: self.__object_to_int(row[column_name], data[column_name].unique()),
            axis=1)

    def __object_to_int(self, object_, object_list):
        xs = sorted([x for x in object_list if x is not np.nan])
        for i, x in enumerate(xs):
            if x == object_:
                return i

        return np.nan

    def data(self):
        return self.__parse_data(self.raw_data)

    def passenger_id(self):
        return self.raw_data['PassengerId']


class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def realpath(relative_path):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, relative_path)


def train(data_train):
    return xgb.XGBClassifier().fit(data_train.X, data_train.y)


def cross_validate(csv_train):
    data_train, data_test = csv_train.train_test_split()
    model = train(data_train)
    predictions = model.predict(data_test.X)
    accuracy = accuracy_score(data_test.y, predictions)
    print('>>> accuracy: {}'.format(accuracy))
    # xgb.plot_tree(model)
    # plt.show()


def predict(csv_train, csv_test, result_filename):
    data_train = csv_train.data()
    data_test = csv_test.data()
    model = train(data_train)
    predictions = model.predict(data_test.X)
    result = pd.DataFrame({
        'PassengerId': csv_test.passenger_id(),
        'Survived': predictions,
    })
    result.to_csv(realpath(result_filename), index=False)


def main():
    csv_train = CSV('data/train.csv')
    csv_test = CSV('data/test.csv')
    cross_validate(csv_train)
    predict(csv_train, csv_test, 'data/submission.csv')


if __name__ == "__main__":
    main()
