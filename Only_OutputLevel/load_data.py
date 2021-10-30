import numpy as np
import h5py
import matplotlib.pyplot as plt
def load_Dataset():
    '''
    x中存的是图片信息，共209张，64*64的RGB图像（可以用pyplot显示出来
    y中存的是标记，1：是猫，0：不是猫
    list_classes中存的是标签的名字
    train_set_x_orig 是一个维度为(m_​​train，num_px，num_px，3）的数组。
    :return:
    '''
    train_dataset = h5py.File('../datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) #(209, 64, 64, 3)
    train_set_y_orig = np.array(train_dataset['train_set_y'][:]) #(209,)
    # train_set_y_orig = train_set_y_orig.reshape([209,1])#确保形状是列向量(209, 1)
    train_set_y_orig = train_set_y_orig.reshape(1,train_set_y_orig.shape[0])  #给的示例程序是保存的行向量

    test_dataset = h5py.File('../datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])#(50, 64, 64, 3)
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])#(50,)
    # test_set_y_orig = test_set_y_orig.reshape(50,1)#确保形状是列向量(50, 1)
    test_set_y_orig = test_set_y_orig.reshape(1,test_set_y_orig.shape[0])#示例程序用的行向量(1,50)

    classes = np.array(test_dataset['list_classes'][:]) #[b'non-cat' b'cat']
    # print(classes)

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes


if __name__ == '__main__':
    train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes =load_Dataset()
    #经过标准化后就不满足plt的输出了
    plt.subplot(1,2,1)
    plt.imshow(train_set_x_orig[2])
    plt.subplot(1, 2, 2)
    plt.imshow(test_set_x_orig[0])
    plt.show()
