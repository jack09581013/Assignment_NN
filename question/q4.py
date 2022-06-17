from answer import nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
X = iris.data
lr = 0.01

encoder = OneHotEncoder()
Y = iris.target.reshape(-1, 1)
encoder.fit(Y)
Y = encoder.transform(Y).toarray()

model = nn.Model(loss='cross_entropy')
model.add(nn.Dense(4, 8, lr=lr))
model.add(nn.ReLU())
model.add(nn.Dense(8, 3, lr=lr))

batch_size = 32

for epoch in range(500):
    for batch in range(0, len(X), batch_size):
        x = X[batch:batch + batch_size, :]
        t = Y[batch:batch + batch_size, :]   # data in one batch

        # Forward and backward your model

# y <= your model prediction, not in one-hot encode

# print(iris.target)
# print(y.reshape(-1))
#
# acc = np.sum(iris.target == y) / len(y)
# print('Accuracy: {:.2f} %'.format(acc*100))
