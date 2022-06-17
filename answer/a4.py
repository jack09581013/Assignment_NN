from answer import nn
import numpy as np
import optimizer as opt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
X = iris.data
optimizer = opt.Adam

encoder = OneHotEncoder()
Y = iris.target.reshape(-1, 1)
encoder.fit(Y)
Y = encoder.transform(Y).toarray()

model = nn.Model(loss='cross_entropy')
model.add(nn.Dense(4, 8, optimizer=optimizer))
model.add(nn.ReLU())
model.add(nn.Dense(8, 3, optimizer=optimizer))

batch_size = 32

for epoch in range(300):
    for batch in range(0, len(X), batch_size):
        x = X[batch:batch + batch_size, :]
        t = Y[batch:batch + batch_size, :]

        y = model.forward(x)
        loss = model.backward(y, t)
        print('loss = {:.3f}'.format(loss))

y = model.predict_class(X)

print(iris.target)
print(y.reshape(-1))

acc = np.sum(iris.target == y) / len(y)
print('Accuracy: {:.2f} %'.format(acc*100))
