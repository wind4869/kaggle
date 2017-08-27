import numpy as np
import pandas as pd
from sklearn import linear_model

__author__ = 'matrix'

USEFUL_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    # 'Embarked'
]

CONVERTED_COLUMNS = [
    'Sex',
    'Embarked'
]


def load(filename):
    data_set = pd.read_csv(filename).fillna(0)
    data_set['Sex'] = np.where(data_set['Sex'] == 'female', 0, 1)
    embarked = data_set['Embarked']
    data_set['Embarked'] = np.where(
        embarked == 'C', 0,
        np.where(embarked == 'Q', 1, 2))
    return data_set


def dump(filename, result):
    result.to_csv(filename, index=False)


def one_hot_encoding(data_set):
    return pd.get_dummies(data_set, columns=CONVERTED_COLUMNS)


def train():
    train_set = load('train.csv')

    y = train_set['Survived']
    X = train_set[USEFUL_COLUMNS]

    model = linear_model.LogisticRegression()
    model.fit(X, y)
    return model


def predict(model):
    test_set = load('test.csv')

    passenger_ids = test_set['PassengerId']
    X = test_set[USEFUL_COLUMNS]

    result = pd.DataFrame()
    result['PassengerId'] = passenger_ids
    result['Survived'] = model.predict(X)
    dump('result.csv', result)


if __name__ == '__main__':
    predict(train())
