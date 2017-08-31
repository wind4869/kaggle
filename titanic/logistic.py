import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection

__author__ = 'matrix'

USEFUL_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch'
]

CONVERTED_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch'
]


def load(filename):
    data_set = pd.read_csv(filename)

    data_set['Sex'] = np.where(data_set['Sex'] == 'female', 0, 1)
    data_set['Age'] = data_set['Age'].fillna(data_set['Age'].mean())
    data_set['Age'] = np.where(data_set['Age'] < 20, 0, 1)

    one_hot_encoding(data_set)

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


def cross_validate():
    train_set = load('train.csv')

    y = train_set['Survived']
    X = train_set[USEFUL_COLUMNS]

    model = linear_model.LogisticRegression()
    print(model_selection.cross_val_score(model, X, y))


if __name__ == '__main__':
    cross_validate()
    predict(train())
