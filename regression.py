import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

y = []
x = []

with open('scores.txt') as fin:
    for line in fin.readlines():
        ln = list(map(lambda x: float(x), line.split()))
        y.append(ln[0])
        x.append(ln[1:6])


x = np.array(x).reshape(-1, 5)
y = np.array(y)


(X_train, X_test, y_train, y_test) = model_selection.train_test_split(x, y, test_size=.2)

model = LinearRegression(
    fit_intercept=False, positive=True).fit(X_train, y_train)
r_sq = model.score(x, y)
print(f'coef: {model.coef_}')
y_predict = model.predict(X_test)
print(f'train_MSE: {mean_squared_error(model.predict(X_train), y_train)}')
print(f'test_MSE: {mean_squared_error(y_predict, y_test)}')


#model_selection.cross_val_score(model, X, Y, scoring='neg_mean_absolute_error')


# for i in range(len(y_predict)):
#    print(y_predict[i], y_test[i])
#print(np.array([1, 2, 3]))
