import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

data = pd.read_fwf('copper-new.txt', sep=" ", header=None, names=["y", "x"])
x = np.array(data['x'])
y = np.array(data['y'])
data.head()

X = x.reshape(-1,1)
linreg = LinearRegression()
linreg.fit(X, y)

print("Slope = "+str(linreg.coef_[0]))
print("Intercept = "+str(linreg.intercept_))

linreg_pred = linreg.predict(X)
mse = mean_squared_error(linreg_pred,y)
print("MSE = "+str(mse))

# Polynomial Degree
lw = 2
x_plot = np.linspace(0, 1000, 101)
X_plot = x_plot[:, np.newaxis]
degree = [1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize=(13,5))
for i in degree:
    model = make_pipeline(PolynomialFeatures(i), Ridge(solver='svd'))
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    poly_pred = model.predict(X)
    mse = round(mean_squared_error(poly_pred,y),2)
    plt.subplot(2,5,i)
    plt.scatter(x, y, color='blue', s=30, marker='o', label="training points")
    plt.plot(x_plot, y_plot, linewidth=lw,c='red')
    plt.title("Degree "+str(i)+" MSE = "+str(mse))
plt.tight_layout()

# Cross-Validation
lambs = np.arange(34, 45, 1).tolist()
cves = []
for i in range(len(lambs)):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    mse = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        poly = make_pipeline(PolynomialFeatures(5), Ridge(solver='svd',alpha=lambs[i]))
        poly.fit(X_train, y_train)
        poly_pred = poly.predict(X_test)
        mse += (1/5)*mean_squared_error(poly_pred,y_test)
    cves.append(round(mse,3))
    print("Lambda = "+str(lambs[i])+": CV Error = "+str(cves[i]))
plt.figure()
plt.plot(lambs,cves)
plt.show()

#Linear vs Poly
test = np.array(400).reshape(-1,1)
lin400_pred = linreg.predict(test)
print("Linear Prediction = "+str(round(lin400_pred[0],3)))

poly = make_pipeline(PolynomialFeatures(5), Ridge(solver='svd',alpha=39))
poly.fit(X, y)
poly400_pred = poly.predict(test)
print("Polynomial RR Prediction = "+str(round(poly400_pred[0],3)))