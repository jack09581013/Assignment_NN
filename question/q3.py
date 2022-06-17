from answer import nn
import answer.optimizer as opt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
X = iris.data
# optimizer = opt.LearningRate
optimizer = opt.Adam

encoder = OneHotEncoder()
Y = iris.target.reshape(-1, 1)
encoder.fit(Y)
Y = encoder.transform(Y).toarray()

model = nn.Model(loss='mse')
model.add(nn.Dense(4, 2, optimizer=optimizer))
model.add(nn.LeakyReLU())
model.add(nn.Dense(2, 2, optimizer=optimizer))
model.add(nn.LeakyReLU())
model.add(nn.Dense(2, 4, optimizer=optimizer))
model.add(nn.LeakyReLU())
model.add(nn.Dense(4, 3, optimizer=optimizer))

batch_size = 32
loss = 1
count = 0
while loss > 0.05:
    for batch in range(0, len(X), batch_size):
        x = X[batch:batch + batch_size, :]
        t = Y[batch:batch + batch_size, :]

        # Forward and backward your model

# y <= your model prediction, not in one-hot encode

# print(iris.target)
# print(y.reshape(-1))
#
# acc = np.sum(iris.target == y) / len(y)
# print('Accuracy: {:.2f} %'.format(acc*100))
