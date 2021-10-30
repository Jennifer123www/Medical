from Load_Datasets import Load_Datasets
import numpy as np
import matplotlib.pyplot as plt
from Normalization2 import normalization
from Predict import predict
from Sigmoid import sigmoid,tanh

def MyModel(X_train,Y_train,X_test,Y_test,classes,learning_rate = 0.001,epoch = 1000,hidden_layers = 10):
    '''
    :param X_train: (12288, 209)(nx,m)
    :param Y_train: (1,50)
    :param X_test: (12288, 50)
    :param Y_test: (1,50)
    :param epoch:轮
    :param learning_rate:学习率，如果太大影响收敛效果，如设置成0.01会发生震荡
    :return:
    '''
    #----------------------初始化-------------------------
    # X_train_temp, Y_train_temp, X_test_temp, Y_test_temp, classes = normalization(X_train, Y_train, X_test, Y_test, classes)
    # X_train, Y_train, X_test, Y_test, classes = normalization(X_train, Y_train, X_test, Y_test, classes)
    #初始化参数
    # print(X_train.shape[1])
    # print(Y_train.shape[0])
    w1 = np.random.randn(hidden_layers,X_train.shape[0])*0.01
    b1 = np.zeros((hidden_layers,1))
    # w1 = np.zeros((hidden_layers,X_train.shape[0]))
    # b1 = 0
    print('w1:',w1.shape,'b1:',b1.shape)
    # print(w1)
    w2 = np.random.randn(Y_train.shape[0],hidden_layers)
    b2 = np.zeros((Y_train.shape[0],1))
    # w2 = np.zeros((Y_train.shape[0], hidden_layers))
    # b2 = 0
    print('w2:',w2.shape,'b2:',b2.shape)
    # #-------------------训练--------------------------------
    m = X_train.shape[1]  #m是样本数量
    print('m:',m)
    costs = []
    for i in range(epoch):
        # 向前计算
        #print(w1.shape,X_train.shape)
        # X_train2, Y_train2, X_test2, Y_test2, classes = normalization(X_train, Y_train, X_test, Y_test, classes)
        Z1 = np.dot(w1, X_train) + b1
        # print(Z1.shape)
        A1 = tanh(Z1)#激活函数
        # print(w2.shape,A1.shape)
        Z2 = np.dot(w2,A1) + b2
        A2 =sigmoid(Z2)
        # print("A2:",A2.shape)
        #损失函数
        # print(Y_train.shape,A2.shape)
        #借鉴集成函数 交叉熵函数# cost = (-1 / m) * np.sum(Y_train * np.log(A2) + (1 - Y_train) * np.log(1-A2))
        logprobs = np.multiply(np.log(A2.T),Y_train) + np.multiply((1-Y_train),np.log(1-A2.T))
        cost = - np.sum(logprobs) / m
        #矩阵压缩
        cost = np.squeeze(cost)

        # 反向传播
        dZ2 = A2 - Y_train
        # print(dZ.shape)
        dW2 = (1 / m) * np.dot(dZ2,A1.T)
        dB2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)

        #dZ1到现在也没看明白 怎么做的
        dZ1 = np.multiply(np.dot(w2.T,dZ2),1-np.power(A1,2))
        dW1 = (1/m) * np.dot(dZ1,X_train.T)
        dB1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
    #
        # 参数更新
        w1 = w1 - learning_rate * dW1
        b1 = b1 - learning_rate * dB1
        w2 = w2 - learning_rate * dW2
        b2 = b2 - learning_rate * dB2
    #
        if i % 100 == 0:
            costs.append(cost)
            print('迭代的次数：',i,'误差值：',cost)
     #-------------------------评估--------------------------------
    #*****训练集********
    Y_prediction_train  = predict(w1,b1,w2,b2,X_train)
    print("训练集的准确性：",(100-np.mean(np.abs(Y_prediction_train-Y_train))*100),'%')
    #****测试集********
    Y_prediction_test = predict(w1,b1,w2,b2,X_test)
    print("测试集的准确性：",(100-np.mean(np.abs(Y_prediction_test-Y_test))*100),'%')
    return costs,learning_rate

if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = Load_Datasets()
    train_set_x, train_set_y, test_set_x, test_set_y, classes = normalization(train_set_x, train_set_y, test_set_x, test_set_y, classes)
    # MyModel(train_set_x, train_set_y, test_set_x, test_set_y)
    costs,learning_rate = MyModel(train_set_x,train_set_y,test_set_x,test_set_y,classes)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch (per hundreds)')
    plt.title('Learning rate ='+str(learning_rate))
    plt.show()