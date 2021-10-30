import numpy as np
from Sigmiod import sigmiod
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction_train = np.zeros((1,m)) #初始化训练集Y的预测值
    # print(w.shape)
    A = sigmiod(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction_train[0,i] = 1 if A[0,i]>0.5 else 0
    # print(Y_prediction_train)
    return Y_prediction_train