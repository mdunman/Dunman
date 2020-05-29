import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('q3.csv',header=None)
y = list(data[54])
print(list(y[0:5]))
data = data.drop(labels=54,axis=1)
print("data shape = "+ str(data.shape))
data.head()

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print(len(X_train)/(len(data)))

# Naive Bayes
p0 = y.count(0) / len(y)
p1 = y.count(1) / len(y)
ps = np.array([p0,p1])
nb = GaussianNB(priors=ps)
nb.fit(X_train, y_train)
round(sum(nb.predict(X_test)==y_test)/len(y_test)*100,2)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
round(sum(lr.predict(X_test)==y_test)/len(y_test)*100,2)

# KNN
print("K Neighbors")
for i in range(2,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print(str(i) +": " + str(round(sum(knn.predict(X_test)==y_test)/len(y_test)*100,2)))
    
#
data2 = np.array(data.loc[:,0:1])
X_train2 = np.array(X_train.loc[:,0:1])
X_test2 = np.array(X_test.loc[:,0:1])
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
x_min, x_max = data2[:, 0].min() - 1, data2[:, 0].max() + 1
y_min, y_max = data2[:, 1].min() - 1, data2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Naive Bayes
p0 = y.count(0) / len(y)
p1 = y.count(1) / len(y)
ps = np.array([p0,p1])
nb2 = GaussianNB(priors=ps)
nb2.fit(X_train2, y_train)

Z = nb2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

ptr = plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, marker='.', s=80, cmap=cmap_bold)
pte = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test, marker='*', s=80, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Naive Bayes")
plt.legend((ptr, pte), ['Training Point', 'Test Point'])
plt.show()

# Logistic Regression
lr2 = LogisticRegression()
lr2.fit(X_train2, y_train)

Z = lr2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

ptr = plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, marker='.', s=80, cmap=cmap_bold)
pte = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test, marker='*', s=80, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Logistic Regression")
plt.legend((ptr, pte), ['Training Point', 'Test Point'])
plt.show()

# KNN
plt.figure(figsize=(11,11))
for i in range(2,11):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(X_train2,y_train)

    Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(3,3,i-1)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    ptr = plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, marker='.', s=50, cmap=cmap_bold)
    pte = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test, marker='*', cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("KNN: " + str(i) + " Neighbors")
    plt.legend((ptr, pte), ['Training Point', 'Test Point'])
plt.tight_layout()

# New Data
data = sio.loadmat('data.mat')['data'].T
print(data.shape)
labels = sio.loadmat('label.mat')['trueLabel'][0]
y = []
for i in range(len(labels)):
    if labels[i] == 6:
        y.append(1)
    if labels[i] == 2:
        y.append(0)
print(labels[0:5])
print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print(len(X_train)/(len(data)))

# Naive Bayes
p0 = y.count(0) / len(y)
p1 = y.count(1) / len(y)
ps = np.array([p0,p1])
nb = GaussianNB(priors=ps)
nb.fit(X_train, y_train)
round(sum(nb.predict(X_test)==y_test)/len(y_test)*100,2)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
round(sum(lr.predict(X_test)==y_test)/len(y_test)*100,2)

# KNN
print("K Neighbors")
for i in range(2,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print(str(i) +": " + str(round(sum(knn.predict(X_test)==y_test)/len(y_test)*100,2)))