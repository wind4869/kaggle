import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

__author__ = 'matrix'

USEFUL_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Cabin'
]

CONVERTED_COLUMNS = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Cabin'
]


def load(filename):
    data_set = pd.read_csv(filename)

    data_set['Sex'] = np.where(data_set['Sex'] == 'female', 0, 1)
    data_set['Age'] = data_set['Age'].fillna(data_set['Age'].mean())
    data_set['Age'] = np.where(
        data_set['Age'] < 10, 0,
        np.where(data_set['Age'] < 60, 1, 2)
    )
    data_set['Cabin'] = np.where(data_set['Cabin'].isnull(), 0, 1)

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

    return [
        SVC().fit(X, y),
        # MultinomialNB().fit(X, y),
        # KNeighborsClassifier(n_neighbors=10).fit(X, y),
        LogisticRegression().fit(X, y)
    ]


def predict(models):
    test_set = load('test.csv')

    passenger_ids = test_set['PassengerId']
    X = test_set[USEFUL_COLUMNS]

    result = pd.DataFrame()
    result['PassengerId'] = passenger_ids

    predicts = [model.predict(X) for model in models]
    result['Survived'] = np.where(sum(predicts) >= len(predicts) / 2 + 1, 1, 0)

    dump('result.csv', result)


def cross_validate():
    train_set = load('train.csv')

    y = train_set['Survived']
    X = train_set[USEFUL_COLUMNS]

    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(scores, scores.mean())

    train_sizes, train_score, test_score = learning_curve(
        model, X, y, cv=10, scoring='accuracy', train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
    )

    plt.style.use('ggplot')
    plt.plot(train_sizes, train_score.mean(axis=1), label='Train')
    plt.plot(train_sizes, test_score.mean(axis=1), label='Test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # cross_validate()
    predict(train())
