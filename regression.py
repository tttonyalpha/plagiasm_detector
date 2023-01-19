import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y = []
x = []

with open('scores.txt') as fin:
    for line in fin.readlines():
        ln = list(map(lambda x: float(x), line.split()))
        y.append(ln[0])
        x.append(ln[1:])

x = np.array(x).reshape(-1, 5)
y = np.array(y)


(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=.2)

model = LinearRegression(
    fit_intercept=False, positive=True).fit(X_train, y_train)
r_sq = model.score(x, y)
print(model.intercept_, model.coef_)
y_predict = model.predict(X_test)
for i in range(len(y_predict)):
    print(y_predict[i], y_test[i])
#print(np.array([1, 2, 3]))
