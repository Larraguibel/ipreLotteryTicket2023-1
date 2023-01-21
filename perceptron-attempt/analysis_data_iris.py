import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron

df = pd.read_csv('iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values


# representar datos

# plt.scatter(X[:50, 0], X[:50, 1], 
#    color="red", marker="o", label="setosa")

# plt.scatter(X[50:100, 0], X[50:100, 1], 
#    color="blue", marker="x", label="versicolor")

# plt.xlabel('sepal lenth [cm]')
# plt.ylabel('petal lenth [cm]')
# plt.legend(loc="upper left")
# plt.show()

ppn = Perceptron(eta = 0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('epochs')
plt.ylabel('number of updates')
plt.show()