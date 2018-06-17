# -*-coding:utf-8-*-

__author__ = 'hll'
__date__ = '2018/6/16 12:31'

import numpy as np

import pandas as pd


from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.svm import SVC


# train_data
train_data = pd.read_csv('train.csv')
train_y = train_data['Survived']
train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1)
train_data['Sex'].replace({'female': 0, 'male': 1}, inplace=True)
train_data['Age'].fillna(np.mean(train_data['Age']), inplace=True)
train_data = scale(train_data)

# test_data
test_data = pd.read_csv('test.csv')
PassengerId = test_data['PassengerId']
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1)

test_data['Sex'].replace({'female': 0, 'male': 1}, inplace=True)
test_data['Age'].fillna(np.mean(test_data['Age']), inplace=True)

test_data.fillna(0, inplace=True)

# lr = LogisticRegression()
svc = SVC()

cs = cross_val_score(svc, train_data, train_y)





test_data = scale(test_data)

svc.fit(train_data, train_y)

Survived = svc.predict(test_data)

df = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Survived})

df.to_csv('results.csv', index=False)