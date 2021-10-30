import numpy as np
from Sigmoid import sigmoid,tanh
def predict(w1,b1,w2,b2,X):
    m = X.shape[1]
    Y_prediction_train = np.zeros((1,m)) #初始化训练集Y的预测值
    # print(w1.shape)
    A0 = tanh(np.dot(w1,X)+b1)
    A = sigmoid(np.dot(w2, A0) + b2)
    for i in range(A.shape[1]):
        Y_prediction_train[0,i] = 1 if A[0,i]>0.5 else 0
    # print(Y_prediction_train)
    return Y_prediction_train