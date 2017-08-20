import math
from sklearn import linear_model
from pandas import read_csv, DataFrame

train = read_csv('train.csv')
target = train['Survived']

data = train
map(data.pop, ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'])
data['Sex'] = map(lambda sex: 0 if sex == 'female' else 1, data['Sex'])
data['Age'] = map(lambda age: 0 if math.isnan(age) else age, data['Age'])

logistic = linear_model.LogisticRegression()
logistic.fit(data.values, target)

test = read_csv('test.csv')
passenger_ids = test['PassengerId']
map(test.pop, ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
test['Sex'] = map(lambda sex: 0 if sex == 'female' else 1, test['Sex'])
test['Age'] = map(lambda age: 0 if math.isnan(age) else age, test['Age'])
test['Fare'] = map(lambda fare: 0 if math.isnan(fare) else fare, test['Fare'])

result = DataFrame()
result['PassengerId'] = passenger_ids
result['Survived'] = logistic.predict(test.values)
result.to_csv('result.csv', index=False)
