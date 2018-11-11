import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
from LoadDataSet import load_dataset
from Model import model
def main():
    #loading the dataset from .h5 files
    #test_catvnoncat.h5
    #train_catvnoncat.h5
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    print(" no of training samples : {0}".format(train_set_y.shape[1]))
    print(" no of testing samples : {0}".format(test_set_y.shape[1]))
    print(" Height and width of an image in the sample : {0} {1} {2}".format(train_set_x_orig.shape[1],train_set_x_orig.shape[2],train_set_x_orig.shape[3]))
    ### reshaping the training and test data set in 
    ### train_set_x_flatten that contains image samples stored as columns and each row is a features 
    train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0], train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3]).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # print(train_set_x_flatten.shape, test_set_x_flatten.shape)
    # print("*****sanity checking *****")
    # print(str(train_set_x_flatten[:140,0]))
    # print(train_set_x_orig)

    #standarding the dataset by taking the mean and subtracting the data by mean and dividing it with the standard deviation.
    #but where the range of the data is same, so here we just divide the data by 255
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 100000, learning_rate = 0.4, print_cost = True)

    print(d["num_iterations"])
    print(d["learning_rate"])

    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
if __name__ == '__main__':
    main()