import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def printing_data():
    x=[]
    y=[]
    path = os.getcwd() + '/data/ex1data1.txt'
    csvfile = open(path, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

    plt.plot(x, y, 'ro')
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.show()


def compute_cost(X, y, theta):
    inner  =np.power(((X * theta.T) - y), 2)
    return np.sum(inner)/(2*len(X))


def debug(X, y, theta):
    print(X.shape)
    print(y.shape)
    print(theta.shape)


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T ) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = compute_cost(X, y, theta)
    return theta, cost


path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# printing_data()
data = (data - data.mean())/data.std()


data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
# debug(X, y, theta)

# print(compute_cost(X, y, theta))
alpha = 0.01
iters = 10000
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(compute_cost(X, y, g))

# printing the results
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
plt.show()

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()