import numpy as np

def least_square(A, b):
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)

number_of_input = 10
number_of_feature = 5
number_of_labels = 2
learning_rate = 0.1

A = np.random.randn(number_of_input, number_of_feature)
b = np.random.randn(number_of_input, number_of_labels)

# Solving x by least square
x = least_square(A, b)
print('least square x:', x.reshape(-1))

# Solving x by gradient decent
x = np.random.randn(number_of_feature, number_of_labels)
print('initial x:', x.reshape(-1))
for epoch in range(500):
    # Write your code here and delete "pass"
    pass

print('neural network x:', x.reshape(-1))


