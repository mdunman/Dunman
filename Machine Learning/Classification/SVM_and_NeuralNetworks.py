import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('q3.csv',header=None)
y = list(data[54])
print(list(y[0:5]))
data = data.drop(labels=54,axis=1)
print("data shape = "+ str(data.shape))
data.head()

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print(len(X_train)/(len(data)))

# Round 1
# SVM
for i in [0.01,0.1,1.00,10.,100]:
    svm = SVC(C=i)
    svm.fit(X_train, y_train)
    p = round(sum(svm.predict(X_test)==y_test)/len(y_test)*100,2)
    print("C = "+str(i)+" -> "+str(p))
    
# NN
nn = MLPClassifier(hidden_layer_sizes=(5,2), activation='logistic', solver='lbfgs', random_state=29, max_iter = 600)
nn.fit(X_train, y_train)
round(sum(nn.predict(X_test)==y_test)/len(y_test)*100,2)

# Round 2
data2 = np.array(data.loc[:,0:1])
X_train2 = np.array(X_train.loc[:,0:1])
X_test2 = np.array(X_test.loc[:,0:1])
h = 0.02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
x_min, x_max = data2[:, 0].min() - 1, data2[:, 0].max() + 1
y_min, y_max = data2[:, 1].min() - 1, data2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# SVM
c = [0.1,1,10,100]
plt.figure(figsize=(12,9))
for i in range(1,5):
    svm2 = SVC(C=c[i-1])
    svm2.fit(X_train2, y_train)
    p = round(sum(svm2.predict(X_test2)==y_test)/len(y_test)*100,2)

    Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(2,2,i)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    ptr = plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, marker='.', s=80, cmap=cmap_bold)
    pte = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test, marker='*', s=80, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM: C = "+str(c[i-1])+" -> "+str(p)+"%")
    plt.legend((ptr, pte), ['Training Point', 'Test Point'])
plt.show()

# NN
nn2 = MLPClassifier(hidden_layer_sizes = (5,2), activation='logistic', solver='lbfgs', random_state=29, max_iter = 600)
nn2.fit(X_train2, y_train)
p = round(sum(nn2.predict(X_test2)==y_test)/len(y_test)*100,2)

Z = nn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

ptr = plt.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train, marker='.', s=80, cmap=cmap_bold)
pte = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test, marker='*', s=80, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Neural Network -> "+str(p)+"%")
plt.legend((ptr, pte), ['Training Point', 'Test Point'])
plt.show()

# Round 3
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

# SVM
for i in [0.01,0.1,1.00,10.,100]:
    svm = SVC(C=i)
    svm.fit(X_train, y_train)
    p = round(sum(svm.predict(X_test)==y_test)/len(y_test)*100,2)
    print("C = "+str(i)+" -> "+str(p))
    
# NN
nn = MLPClassifier(hidden_layer_sizes=(5,2), activation='logistic', solver='lbfgs', random_state=29, max_iter = 600)
nn.fit(X_train, y_train)
round(sum(nn.predict(X_test)==y_test)/len(y_test)*100,2)