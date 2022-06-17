import torch.nn
from answer.torch_model import IRIS
from torch.optim import Adam
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
X = torch.from_numpy(iris.data).float()

encoder = OneHotEncoder()
Y = iris.target.reshape(-1, 1)
encoder.fit(Y)
Y = torch.from_numpy(encoder.transform(Y).toarray()).float()

model = IRIS()
optimizer = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
mse_loss = torch.nn.MSELoss()

batch_size = 32
count = 0
while count < 1000:
    for batch in range(0, len(X), batch_size):
        optimizer.zero_grad()
        x = X[batch:batch + batch_size, :]
        t = Y[batch:batch + batch_size, :]

        y = model(x)
        loss = mse_loss(y, t)
        print('loss = {:.3f}'.format(loss))
        loss.backward()
        optimizer.step()
    count += 1

y = model.predict_class(X).numpy()
print(y)

print('count =', count)
print(iris.target)
print(y.reshape(-1))

acc = np.sum(iris.target == y) / len(y)
print('Accuracy: {:.2f} %'.format(acc*100))
