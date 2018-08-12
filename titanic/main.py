#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb


class CSV:
    def __init__(self, filename):
        self.raw_data = pd.read_csv(realpath(filename))

    def __parse_data(self, raw_data):
        data = raw_data

        # Title
        data['Title'] = data['Name'].apply(self.__get_title)

        # Age
        data['Age'] = data.apply(self.__get_age, axis=1)

        # HasCabin
        data['HasCabin'] = data['Cabin'].apply(
            lambda x: 1 if pd.notna(x) else 0)

        # 类型整理
        self.__column_object_to_int(data, 'Sex')
        self.__column_object_to_int(data, 'Embarked')
        self.__column_object_to_int(data, 'Title')

        X = data[[
            'Pclass',
            'Sex',
            'Age',
            'HasCabin',
            'Embarked',
            'Fare',
            'SibSp',
            'Parch',
        ]]
        y = data.get('Survived')
        return Data(X, y)

    def __get_title(self, name):
        title = name.split(',')[1].split('.')[0].strip()
        if title in ['Mlle', 'Lady', 'the Countess', 'Ms', 'Mme']:
            return 'Mrs'

        if title in ['Rev', 'Col', 'Major', 'Sir', 'Jonkheer', 'Capt', 'Don']:
            return 'Mr'

        return title

    def __get_age(self, row):
        if pd.notna(row['Age']):
            return row['Age']

        if row['Title'] == 'Master':
            return 5  # 小男孩

        if row['Title'] == 'Miss' and row['Parch'] > 0:
            return 5  # 小女孩

        return 20  # 成年人

    def __fillna_median(self, raw_data, column_name):
        raw_data[column_name] = raw_data[column_name].fillna(
            raw_data[column_name].median())

    def __fillna_most_frequent(self, raw_data, column_name):
        most_frequent_value = raw_data.groupby(column_name)[
            'PassengerId'].nunique().idxmax()
        raw_data[column_name] = raw_data[column_name].fillna(
            most_frequent_value)

    def __column_object_to_int(self, data, column_name):
        data[column_name] = data[column_name].apply(
            lambda x: self.__object_to_int(x, data[column_name].unique()))

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


def grid_search_cv(csv_train):
    data_train = csv_train.data()
    params = {
        'n_estimators': np.logspace(1, 3, num=10).astype(int),
        'max_depth': range(2, 6, 1),
    }
    # params = {
    #     'n_estimators': [359],
    #     'max_depth': [2],
    # }
    model = xgb.XGBClassifier(n_jobs=4)
    cv = GridSearchCV(
        model, params, scoring='accuracy', cv=6).fit(data_train.X,
                                                     data_train.y)
    print('>>> cv.best_params: {}, cv.best_score_: {}\n'.format(
        cv.best_params_, cv.best_score_))
    model = cv.best_estimator_
    xgb.plot_tree(model)
    plt.show()
    return model


def predict(model, csv_test, result_filename):
    data_test = csv_test.data()
    predictions = model.predict(data_test.X)
    result = pd.DataFrame({
        'PassengerId': csv_test.passenger_id(),
        'Survived': predictions,
    })
    result.to_csv(realpath(result_filename), index=False)


def main():
    csv_train = CSV('data/train.csv')
    csv_test = CSV('data/test.csv')
    model = grid_search_cv(csv_train)
    predict(model, csv_test, 'data/submission.csv')


if __name__ == "__main__":
    main()
