#!/usr/bin/env python3

import xgboost as xgb


def main():
    data_train = xgb.DMatrix('data/train.csv')
    print('>>> data_train: {}'.format(data_train))


if __name__ == "__main__":
    main()
