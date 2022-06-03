# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import scipy.io as scio
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 8))


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z

# return w_hj* for enhancement nodes
def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def confusion_matrix(baseline, result):

    # make the pyramid square
    max_len = max((len(l) for l in result))
    result = list(map(lambda l:l + [0]*(max_len - len(l)), result))
    
    result=torch.Tensor(result)
    baseline=torch.Tensor(baseline)

    nt = result.size(0)
    # acc[t] equals result[t,t]
    acc = result.diag()  # current acc
    fin = result[nt - 1] # final acc
    # bwt[t] equals result[T,t] - acc[t]
    bwt = fin - acc
    # fwt[t] equals result[t,t] - baseline[t]
    fwt = acc - baseline
    # # fwt[t] equals result[t-1,t] - baseline[t]
    # fwt = torch.zeros(nt)
    # for t in range(1, nt):
    #     fwt[t] = result[t - 1, t] - baseline[t]
    return fin.mean(), bwt[0:(nt-1)].mean(), fwt[1:nt].mean()


# train process
def train(train_x, train_y, s, c, N1, N2, N3):
    # L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    train_x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    Z = np.zeros([train_x.shape[0], N2 * N1])
    
    We_set = []  # save We_star
    max_min = []  # distance of max and min value w.r.t. Zi
    min_value = [] # min value w.r.t. Zi
    
    time_start = time.time()

    # feature nodes
    for i in range(N2):  # generate feature nodes with N2 windows
        random.seed(i)  # fix random assignment for each window
        We = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        X_We_B = np.dot(train_x_bias, We)  # feature of each window, alternative to X_We_B=(X*We+B)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_We_B)
        feature1 = scaler1.transform(X_We_B)  # feature of each window after preprocessing
        We_star = sparse_bls(feature1, train_x_bias).T  # to obtain We_star
        We_set.append(We_star)
        Zi = np.dot(train_x_bias, We_star)  # equal to (X*We_star+B)
        # print('Feature nodes in window: max:',np.max(Zi),'min:',np.min(Zi))
        max_min.append(np.max(Zi, axis=0) - np.min(Zi, axis=0))
        min_value.append(np.min(Zi, axis=0))
        Zi = (Zi - min_value[i]) / max_min[i]
        Z[:, N1 * i:N1 * (i + 1)] = Zi
        del Zi
        del X_We_B
        del We
    
    # enhancement nodes
    Z_bias = np.hstack([Z, 0.1 * np.ones((Z.shape[0], 1))])

    if N1 * N2 >= N3:
        # random.seed(67797325)
        random.seed()
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        # random.seed(67797325)
        random.seed()
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    Z_Wh_B = np.dot(Z_bias, Wh)
    # print('Enhance nodes: max:',np.max(Z_Wh_B),'min:',np.min(Z_Wh_B))
    param_shrink = s / np.max(Z_Wh_B)
    H = tansig(Z_Wh_B * param_shrink)
    A = np.hstack([Z, H])  # the expended input matrix
    A_p = pinv(A, c)
    OutputWeight = np.dot(A_p, train_y)

    time_end = time.time()
    train_time = time_end - time_start

    train_output = np.dot(A, OutputWeight)
    trainAcc = show_accuracy(train_output, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    # print('Training time is ', train_time, 's')

    return We_set, min_value,max_min, Wh, param_shrink, OutputWeight
   

# test process
def test(test_x, test_y, N1, N2, We_set, min_value, max_min, Wh, param_shrink, OutputWeight):
    ymin = 0
    ymax = 1
    test_x = preprocessing.scale(test_x, axis=1)
    test_x_bias = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    Z_test = np.zeros([test_x.shape[0], N2 * N1])
    
    time_start = time.time()

    for i in range(N2):
        X_We_B = np.dot(test_x_bias, We_set[i])  # feature of each window for test, alternative to test_x_We_B=(X*We+B)
        Z_test[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (X_We_B - min_value[i]) / max_min[i] - ymin  # anti-normalization

    Z_test_bias = np.hstack([Z_test, 0.1 * np.ones((Z_test.shape[0], 1))])
    Z_test_Wh_B = np.dot(Z_test_bias, Wh)

    H_test = tansig(Z_test_Wh_B * param_shrink)

    A_test = np.hstack([Z_test, H_test])

    test_output = np.dot(A_test, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(test_output, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    # print('Testing time is ', testTime, 's')
    return testAcc


# Implementation details for BLS
def main():

    # Class-IL Scenario: FashionMNIST-10/5
    dataFile = 'FashionMNIST_10_5\FashionMNIST_split5_CIL.mat'
    data = scio.loadmat(dataFile)

    N1 = 10     # of nodes belong to each window
    N2 = 10     # of windows -------Feature mapping layer
    N3 = 1500    # of enhancement nodes -----Enhance layer
    # L = 5       # of incremental steps
    # M1 = 50     # of adding enhance nodes
    s = 0.8     # shrink coefficient of the enhancement nodes
    C = 2**-30  # Regularization coefficient
    TN = 5 # the length of sequentical tasks
    R = [] # record all the test acc
    Ri = [] # test acc of all tasks after learning i-th task
    task_index=0 # the position in sequentical tasks

    time_start = time.time()
    print('-------------------BLS---------------------------')
    random.seed() # to guarantee the task_orders different per run
    # task_orders = [1,2,3,4,5]
    task_orders = np.random.permutation(TN)+1
    print('Command: train on current task and then test on seen ones')
    print('#'*8,'random task order of current run:', task_orders)
    # first learn task i
    for i in task_orders:
        task_index += 1 # the training needs FIM when task_num > 1
        train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
        train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
        We_set, min_value,max_min, Wh, param_shrink, OutputWeight=train(train_x, train_y, s, C, N1, N2, N3)

        # then test the seen tasks 
        for j in task_orders[0:task_index]:
            test_x = np.double(data['test_x_'+str((j-1)*2)+str(j*2-1)])
            test_y = np.double(data['test_y_'+str((j-1)*2)+str(j*2-1)])
            test_accij = test(test_x, test_y, N1, N2, We_set, min_value, max_min, Wh, param_shrink, OutputWeight) # test on task j
            Ri.append(test_accij) 
        R.append(Ri)
        Ri = []

    time_end = time.time()
    all_performed_time = time_end - time_start
    print('Accumulative Training time on {} tasks  is {:.4f} s'.format(task_index, all_performed_time))
    print('Average accuracy on {} tasks  is {:.4f}%'.format(task_index, np.mean(R[4]) * 100))

    return np.mean(R[4]), all_performed_time
    
    
    # # obtain the classification accuracy of an independent model trained only on each task.   
    # print('Independent model trained only on each task...')
    # baseline=[]
    # # train and test task i by the above task order
    # for i in task_orders:
    #     task_index = 1 # be seperatedly trained without consolidation
    #     train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
    #     train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
    #     test_x = np.double(data['test_x_'+str((i-1)*2)+str(i*2-1)])
    #     test_y = np.double(data['test_y_'+str((i-1)*2)+str(i*2-1)])
    #     We_set, min_value,max_min, Wh, param_shrink, OutputWeight=train(train_x, train_y, s, C, N1, N2, N3) 
    #     test_acci = test(test_x, test_y, N1, N2, We_set, min_value, max_min, Wh, param_shrink, OutputWeight) # test on task j 
    #     baseline.append(test_acci)

    # acc, bwt, fwt = confusion_matrix(baseline, R)
    # return acc, bwt, fwt, all_performed_time
    

if __name__ == '__main__':
    
    Multiple=10 # multiple runs for the mean and std of metrics
    
    # ACC =torch.Tensor([])
    # BWT=torch.Tensor([])
    # FWT=torch.Tensor([])
    # Time=torch.Tensor([])
    ACC =[]
    # BWT=[]
    # FWT=[]
    Time=[]
    for multi_runs in range(Multiple):
        # acc, bwt, fwt, all_performed_time=main()
        acc, all_performed_time=main()
        ACC.append(acc)
        # BWT.append(bwt)
        # FWT.append(fwt)
        Time.append(all_performed_time)
    print('Results of {} repeated runs'.format(Multiple))
    print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)))
    # print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)))
    # print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)))
    print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)))

    # also saved in the files
    method = 'BLS'
    fname= method
    fname += datetime.datetime.now().strftime("_%m_%d_%H_%M")  # "%Y_%m_%d_%H_%M_%S"
    fname = os.path.join('FashionMNIST_10_5\Results', fname + '.txt')
    if fname is not None:
        f = open(fname, 'w')
        print('Results of runing {} with {} randomly shuffled task orders'.format(method ,Multiple), file=f)
        print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)), file=f)
        # print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)), file=f)
        # print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)), file=f)
        print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)), file=f)
        f.close()