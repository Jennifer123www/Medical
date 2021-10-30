from Load_Datasets import Load_Datasets
import numpy as np
def normalization(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes,disrupt_the_order = True):
    # # 程序并没有直接使用原来的数据，而是经过了处理,标准化，使像素值都落在【0,1】
    m_train = train_set_x_orig.shape[0] #训练集中样本的数量
    # num_px = train_set_x_orig.shape[1] #训练集中样本的像素
    m_test = test_set_x_orig.shape[0] #测试集中样本的数量
    # print(m_train,num_px,m_test) #209 64 50

    #将训练集、测试集的维度降低并转置。
    # print(train_set_x_orig.shape,train_set_y_orig.shape)
    permutation_train = list(np.random.permutation(m_train))
    shuffled_train_x = train_set_x_orig[permutation_train,:,:,:]
    shuffled_train_y = train_set_y_orig[:,permutation_train].reshape((1,m_train))
    # print(shuffled_train_x.shape, shuffled_train_y.shape)

    # permutation_test = list(np.random.permutation(m_test))
    # shuffled_test_x = test_set_x_orig[permutation_test,:,:,:]
    # shuffled_test_y = test_set_y_orig[:,permutation_test].reshape((1,m_test))

    # np.random.shuffle(train_set_x_orig)
    # np.random.shuffle(test_set_x_orig)


    train_set_x_flatten = train_set_x_orig.reshape(shuffled_train_x.shape[0],-1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
    # print(train_set_x_flatten.shape,test_set_x_flatten.shape) #(12288, 209) (12288, 50)

    #标准化我们的数据集,除以255，让标准化的数据位于 [0,1] 之间
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    # print(train_set_x.shape,test_set_x.shape) #(12288, 209) (12288, 50))

    return train_set_x,shuffled_train_y,test_set_x,test_set_y_orig,classes
if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = Load_Datasets()
    X_train2, Y_train2, X_test2, Y_test2, classes = normalization(train_set_x, train_set_y, test_set_x, test_set_y, classes)