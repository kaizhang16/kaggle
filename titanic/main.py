#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


class Data:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def object_to_int(object_, object_list):
    for i, x in enumerate(object_list):
        if x == object_:
            return i

    return -1


def object_column_to_int(data, column_name):
    data[column_name] = data.apply(
        lambda row: object_to_int(row[column_name], data[column_name].unique()),
        axis=1)


def get_X_y(raw_data):
    raw_data = raw_data.drop(['Name', 'Ticket'], axis=1)
    object_column_to_int(raw_data, 'Sex')
    object_column_to_int(raw_data, 'Cabin')
    object_column_to_int(raw_data, 'Embarked')
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


def main():
    data_train, data_test = load_train_test('data/train.csv')
    # print('>>> data_train.X:\n{}'.format(data_train.X))
    # print('>>> data_train.y:\n{}'.format(data_train.y))
    # print('>>> data_test.X:\n{}'.format(data_test.X))
    model = train(data_train)
    predictions = model.predict(data_test.X)
    # print('>>> predictions: {}'.format(predictions))
    right = 0
    for i, p in enumerate(predictions):
        if p == data_test.y.iloc[i]:
            right += 1
    accuracy = right / len(data_test.y)
    print('>>> accuracy: {}'.format(accuracy))


if __name__ == "__main__":
    main()
