from Only_OutputLevel.load_data import load_Dataset
import numpy as np
import matplotlib.pyplot as plt
from Normalization import normalization
from predict import predict
from Sigmiod import sigmiod

def MyModel(X_train,Y_train,X_test,Y_test,classes,learning_rate = 0.001,epoch = 1000):
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
    X_train_temp, Y_train_temp, X_test_temp, Y_test_temp, classes = normalization(X_train, Y_train, X_test, Y_test,classes)
    #初始化参数
    w = np.zeros((X_train_temp.shape[0],1))#应该都初始化为（nx,1） 因为后续操作 w叉乘x
    b = 0         #b初始化为0就行，之后利用广播机制
    print('w:',w.shape,'b:',b)
    #-------------------训练--------------------------------
    m = X_train.shape[1]  #应该是样本数量
    print(m)
    costs = []
    for i in range(epoch):
        X_train2, Y_train2, X_test2, Y_test2, classes = normalization(X_train, Y_train, X_test, Y_test, classes)
        # 向前计算
        # print(X_train.shape)
        Z = np.dot(w.T, X_train2) + b
        # print(Z.shape)
        A = sigmiod(Z)#激活函数
        #损失函数
        cost = (-1 / m) * np.sum(Y_train2 * np.log(A) + (1 - Y_train2) * np.log(1-A))
        #矩阵压缩
        cost = np.squeeze(cost)
        # 反向传播
        dZ = A - Y_train2
        # print(dZ.shape)
        dW = (1 / m) * np.dot(X_train2,dZ.T)
        dB = (1 / m) * np.sum(dZ)

        # 参数更新
        w = w - learning_rate * dW
        b = b - learning_rate * dB

        if i % 100 == 0:
            costs.append(cost)
            print('迭代的次数：',i,'误差值：',cost)
     #-------------------------评估--------------------------------
    #*****训练集********
    Y_prediction_train  = predict(w,b,X_train_temp)
    print("训练集的准确性：",(100-np.mean(np.abs(Y_prediction_train-Y_train_temp))*100),'%')
    #****测试集********
    Y_prediction_test = predict(w,b,X_test_temp)
    print("测试集的准确性：",(100-np.mean(np.abs(Y_prediction_test-Y_test_temp))*100),'%')
    return costs,learning_rate

if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_Dataset()
    # train_set_x, train_set_y, test_set_x, test_set_y, classes = normalization(train_set_x, train_set_y, test_set_x, test_set_y, classes)
    costs,learning_rate = MyModel(train_set_x,train_set_y,test_set_x,test_set_y,classes)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epoch (per hundreds)')
    plt.title('Learning rate ='+str(learning_rate))
    plt.show()