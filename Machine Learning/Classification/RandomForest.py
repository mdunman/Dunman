import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

data = pd.read_csv('spambase-data.csv',header=None)
y = np.array(data[57])
data = data.drop(labels=57,axis=1)
data.head()

print("# not spam: "+str(len(list(y[y==0]))))
print("# spam: "+str(len(list(y[y==1]))))
print("# total: "+str(len(y)))
print("# features: "+str(data.shape[1]))

m,n = data.shape
adata = np.array(data)
for i in range(m):
    for j in range(n):
        if np.isnan(adata[i][j])==True:
            adata[i][j]=0
            
ndata = preprocessing.scale(adata)
ndata = pd.DataFrame(ndata)
ndata.head()

# Decision Tree
oak = tree.DecisionTreeClassifier(random_state=29)
oak = oak.fit(ndata, y)

plt.figure(figsize=(14,10))
tree.plot_tree(oak, filled=True)
plt.show()

plt.figure(figsize=(14,10))
tree.plot_tree(oak, filled=True, max_depth=2)
plt.show()

# Random Forest
X_train, X_test, y_train, y_test = train_test_split(ndata, y, test_size=0.2)
print(round(len(X_train)/(len(ndata)),2))

depths = list(range(1,41,1))
oak_auc = []
rf_auc = []
for i in range(len(depths)):
    oak = tree.DecisionTreeClassifier(max_depth=depths[i], random_state=29)
    oak = oak.fit(X_train,y_train)
    oak_pred = oak.predict(X_test)
    oak_probs = oak.predict_proba(X_test)[:, 1]
    oak_auc.append(round(roc_auc_score(y_test, oak_probs)*100,2))     
    rf = RandomForestClassifier(max_depth=depths[i], random_state=29)
    rf.fit(X_train,y_train)
    rf_pred = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc.append(round(roc_auc_score(y_test, rf_probs)*100,2))    
    
plt.figure(figsize=(9,6))
plt.scatter(depths,rf_auc,c='b',label='Random Forest')
plt.scatter(depths,oak_auc,c='r',label='Decision Tree')
plt.xlabel("Tree Size")
plt.ylabel("AUC")
plt.title('Random Forest vs Decision Tree Testing AUC')
plt.legend(loc='lower right')
plt.show()