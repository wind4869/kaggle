import math
from sklearn import linear_model
from pandas import read_csv, DataFrame

train = read_csv('train.csv')
target = train['Survived']

data = train
[data.pop(column) for column in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked']]
data['Sex'] = list(map(lambda sex: 0 if sex == 'female' else 1, data['Sex']))
data['Age'] = list(map(lambda age: 0 if math.isnan(age) else age, data['Age']))

logistic = linear_model.LogisticRegression()
logistic.fit(data.values, target)

test = read_csv('test.csv')
passenger_ids = test['PassengerId']
[test.pop(column) for column in ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']]
test['Sex'] = list(map(lambda sex: 0 if sex == 'female' else 1, test['Sex']))
test['Age'] = list(map(lambda age: 0 if math.isnan(age) else age, test['Age']))
test['Fare'] = list(map(lambda fare: 0 if math.isnan(fare) else fare, test['Fare']))

result = DataFrame()
result['PassengerId'] = passenger_ids
result['Survived'] = logistic.predict(test.values)
result.to_csv('result.csv', index=False)
