import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import networkx as nx

#Load Edges 
f_path = abspath("edges.txt")
if exists(f_path):
    with open(f_path) as graph_file:
        lines = [line.split() for line in graph_file]
a = np.array(lines).astype(int)
a[0:5]

#Load Nodes
f_path = abspath("nodes.txt")
idx2name = []
idx2label = []
idx2node = []
if exists(f_path):
    with open(f_path) as fid:
        for line in fid.readlines():
            name = line.split("\t", 2)[1]
            idx2name.append(name[:-1])
            node = line.split("\t", 2)[0]
            idx2node.append(int(node))
            label = line.split("\t", 2)[2][0]
            #label = label.split("\t", 1)
            idx2label.append(label)
print(idx2name[0:5])
print(idx2node[0:5])
print(idx2label[0:5])

G = nx.Graph()
G.add_edges_from(a.tolist())
G.number_of_edges()
G.number_of_nodes()

(G.subgraph(c) for c in nx.connected_components(G))
nx.connected_components(G)
S=[G.subgraph(c).copy() for c in nx.connected_components(G)]
list(nx.connected_components(G))

nodes = []
names = []
labels = []
remove = []
for i in idx2node:
    if i in a:
        nodes.append(i)
        names.append(idx2name[i-1])
        labels.append(idx2label[i-1])
    else:
        remove.append(i)
print(nodes[0:5])
print(names[0:5])
print(labels[0:5])
print(remove[0:5])

new_nodes = list(range(1,len(nodes)+1))
new_nodes[0:5]

for i in range(0,len(a)):
    a[i][0] = nodes.index(a[i][0])+1
    a[i][1] = nodes.index(a[i][1])+1
a[0:5]

# spectral clustering
n = len(new_nodes)
k = 2

i = a[:, 0]-1
j = a[:, 1]-1
v = np.ones((a.shape[0], 1)).flatten()

A = sparse.coo_matrix((v, (i, j)), shape=(n, n))
A = (A + np.transpose(A))/2
print(A)

D = np.diag(1/np.sqrt(np.sum(A, axis=1)).A1)
        
L = D @ A @ D
L

v, x = np.linalg.eig(L)
x = x[:, 0:k].real
x = x/np.repeat(np.sqrt(np.sum(x*x, axis=1).reshape(-1, 1)), k, axis=1)

# scatter
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# k-means
kmeans = KMeans(n_clusters=k).fit(x)
c_idx = kmeans.labels_

# show cluster
for i in range(2):
    print(f'Cluster {i+1}\n***************')
    idx = [index for index, t in enumerate(c_idx) if t == i]
    for index in idx:
        print(labels[index])
    print('\n')
    
    