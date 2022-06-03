'''
The codes are proprietary. Please do not distribute and only for this double-blind review.
'''
# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import scipy.io as scio
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import time
import torch
import torch.nn.functional as F
import os
import datetime


# Define classification accuracy
def classification_accuracy(predict_label, Label):
    count = 0
    label = Label.argmax(axis=1)
    prediction = predict_label.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label[j] == prediction[j]:
            count += 1
    return round(count / len(Label), 8)


# Choice of ommonly uesed activation functions
def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1
# def tanh(data):
#     return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))

def relu(data):
    return np.maximum(data, 0)


def pseudo_inv(A, reg):  # reg: regularization coefficient for ill condition
    # A_p = np.mat(A.T.dot(A) + reg * np.eye(A.shape[1])).I.dot(A.T)
    A_p = np.linalg.pinv(A.T.dot(A)).dot(A.T)
    return np.array(A_p)


# Construct the unsupervised sparse autoencoder
def autoencoder(A, b):
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

def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


# preprocess the original input X
def preprocess(train_x):

    N1 = 10     # of nodes belong to each window
    N2 = 10     # of windows -------Feature mapping layer
    N3 = 670    # of enhancement nodes -----Enhancement layer
    s = 300     # shrink coefficient of the enhancement nodes

    train_x = preprocessing.scale(train_x, axis=1)
    train_x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    Z = np.zeros([train_x.shape[0], N2 * N1])
    We_set = []  # save We_star
    max_min = []  # distance of max and min value w.r.t. Zi
    min_value = [] # min value w.r.t. Zi
    # time_start = time.time()

    # feature nodes
    for i in range(N2):  # generate feature nodes with N2 windows
        random.seed(i)  # fix random assignment for each window in training and test
        We = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        X_We_B = np.dot(train_x_bias, We)  # feature of each window, alternative to X_We_B=(X*We+B)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_We_B)
        feature1 = scaler1.transform(X_We_B)  # feature of each window after preprocessing
        We_star = autoencoder(feature1, train_x_bias).T  # to obtain We_star
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
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    Z_Wh_B = np.dot(Z_bias, Wh)
    # print('Enhance nodes: max:',np.max(Z_Wh_B),'min:',np.min(Z_Wh_B))
    param_shrink = s / np.max(Z_Wh_B)
    H = tansig(Z_Wh_B * param_shrink)
    A = np.hstack([Z, H])  # the expended input matrix

    return Z, H, A


# computing meta-plasiticy matrix 
def log_liklihoods(OutputWeight, A, train_y):
    OutputWeight = torch.from_numpy(OutputWeight)
    OutputWeight.requires_grad = True
    A = torch.from_numpy(A)
    output = torch.mm(A, OutputWeight)
    label = torch.from_numpy(train_y)
    log_liklihoods = F.cross_entropy(output, label.max(1)[1])
    log_liklihoods.backward()
    FIM = OutputWeight.grad ** 2    
    return FIM.numpy() * train_y.shape[0]

# computing meta-plasiticy matrix --CUDA is used
# def log_liklihoods(OutputWeight, A, train_y):
#     OutputWeight = torch.from_numpy(OutputWeight).cuda()
#     OutputWeight.requires_grad = True
#     A = torch.from_numpy(A).cuda()
#     output = torch.mm(A, OutputWeight)
#     label = torch.from_numpy(train_y).cuda()
#     log_liklihoods = F.cross_entropy(output, label.max(1)[1])
#     log_liklihoods.backward()
#     FIM = OutputWeight.grad ** 2    
#     return FIM.detach().cpu().numpy() * train_y.shape[0]  # to mitigate value

# For obtaining the metrics 
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
    

# Training process
def train(train_x, train_y, dl, C, Lambda, Lmax, task_index, FIM, lamb, OutputWeight_ed):
    
    time_start = time.time()
    # The output of expanded input layer
    Z, H, A= preprocess(train_x) # return mapping features
    train_x = A # N by N1*21+N3
    
    # Hidden layer
    np.random.seed(2505)  # candicate: 2505/8880/999/10  to guarantee the same random params for all tasks 
    InputWeight = Lambda * (2 * np.random.rand(Lmax, train_x.shape[1]) - 1)
    InputBias = Lambda * (2 * np.random.rand(Lmax, 1) - 1)
    tempH = np.dot(InputWeight, train_x.T) + InputBias
    H = relu(tempH.T)

    if dl == 1:
        A = np.hstack([train_x, H])
    else:
        A = H

    if task_index == 1:
        A_p = pseudo_inv(A, C)  # calculate Pseudo inverse
        OutputWeight = np.dot(A_p, train_y)
    else:
        # consolidate outpu weight with the PDM method
        L = A.shape[1]
        OutputWeight = np.empty((L, 0))
        for q in np.arange(train_y.shape[1]):
            sum_lambF = np.zeros((L, L))
            for t in range(1, task_index):
                FIM_q = np.diag(FIM[t-1][:, q])  # L by L
                sum_lambF += lamb[t-1] * FIM_q
            beta_q = pseudo_inv(A.T.dot(A) + sum_lambF, C).dot(A.T.dot(train_y[:,q:q+1]) + sum_lambF.dot(OutputWeight_ed[:,q:q+1]))
            OutputWeight = np.concatenate((OutputWeight, beta_q), axis=1)

    train_output = np.dot(A, OutputWeight)  # [N, m]
    train_acc = classification_accuracy(train_output, train_y)
    time_end = time.time()
    train_time = time_end - time_start
    
    print('Training accuracy on task {} is {:.4f} %'.format(task_index, train_acc * 100))
    # print('Training time on task {} is {:.4f} s'.format(task_index, train_time)) 
    return InputWeight, InputBias, OutputWeight, A


# Test process
def test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, task_index):
    
    time_start = time.time()
    Z, H, A= preprocess(test_x) 
    test_x = A

    tempH_test = np.dot(InputWeight, test_x.T) + InputBias
    H_test = relu(tempH_test.T)
    if dl == 1:
        A_test = np.hstack([test_x, H_test])
    else:
        A_test = H_test

    test_output = np.dot(A_test, OutputWeight)
    test_acc = classification_accuracy(test_output, test_y)
    time_end = time.time()
    test_time = time_end - time_start
    print('Test accuracy on task_{} is {:.4f}%'.format(task_index, test_acc * 100))
    # print('Test time on {} is {}s'.format(task_index, test_time))  
    return test_acc 
  

# Implementation details for MRNet in the CIL Scenario: FashionMNIST-10/5
def main():

    # Dataloader in one epoch setting
    dataFile = 'FashionMNIST_10_5\FashionMNIST_split5_CIL.mat' # Optionally, one can use datasets.FashionMNIST 
    data = scio.loadmat(dataFile)
    
    # parameter setting
    dl = 0  # without direct links
    Lambda = 1  # the assignment scope of random parameters
    Lmax = 900  # the maximum number of hidden nodes
    C = 2 ** -30  
    TN = 5 # the length of sequentical tasks
    lamb =  [5e3] * (TN-1) # lamb = [3000, 5000, 5000, 6000]
    FIM = []
    OutputWeight = []
    R = [] # record all the test acc
    Ri = [] # test acc of all tasks after learning i-th task
    task_index=0 # the position in sequentical tasks

    time_start = time.time()
    print('-------------------MRNet---------------------------')
    random.seed() # to guarantee the task_orders different per run
    # task_orders = [1,2,3,4,5]
    task_orders = np.random.permutation(TN)+1
    print('Command: train on current task and then test on seen ones')
    print('#'*8,'random task order of current run:', task_orders)
    # first learn task i
    for i in task_orders:
        task_index += 1 # the training needs FIM when task_index > 1
        train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
        train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
        InputWeight, InputBias, OutputWeight, G = train(train_x, train_y, dl, C, Lambda, Lmax, task_index, FIM, lamb, OutputWeight)  # train on task i
        
        # then test the seen tasks 
        for j in task_orders[0:task_index]:
            test_x = np.double(data['test_x_'+str((j-1)*2)+str(j*2-1)])
            test_y = np.double(data['test_y_'+str((j-1)*2)+str(j*2-1)])
            test_accij = test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, j)  # test on task j, R_ij
            Ri.append(test_accij) 
        R.append(Ri)
        Ri = []
        if task_index != TN: # FIM only for (TN-1) tasks
            FIM_i = log_liklihoods(OutputWeight, G, train_y)
            FIM.append(FIM_i)  # consolidation strength matrix
    
    time_end = time.time()
    all_performed_time = time_end - time_start
    print('Accumulative Training and test time on {} tasks  is {:.4f} s'.format(task_index, all_performed_time))
    print('Average accuracy on {} tasks  is {:.4f}%'.format(task_index, np.mean(R[4]) * 100))
    

    # obtain the classification accuracy of an independent model trained only on each task.
    print('Independent model trained only on each task...')   
    baseline=[]
    # train and test task i by the above task order
    for i in task_orders:
        task_index = 1 # be seperatedly trained without consolidation
        train_x = np.double(data['train_x_'+str((i-1)*2)+str(i*2-1)])
        train_y = np.double(data['train_y_'+str((i-1)*2)+str(i*2-1)])
        test_x = np.double(data['test_x_'+str((i-1)*2)+str(i*2-1)])
        test_y = np.double(data['test_y_'+str((i-1)*2)+str(i*2-1)])
        InputWeight, InputBias, OutputWeight, G = train(train_x, train_y, dl, C, Lambda, Lmax, task_index, FIM, lamb, OutputWeight) 
        test_acci = test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, i)  
        baseline.append(test_acci)
        
    acc, bwt, fwt = confusion_matrix(baseline, R)
    return acc, bwt, fwt, all_performed_time


if __name__ == '__main__':
    
    Multiple=10 # multiple runs for the mean and std of metrics
    
    ACC =[]
    BWT=[]
    FWT=[]
    Time=[]
    for multi_runs in range(Multiple):
        acc, bwt, fwt, all_performed_time=main()
        ACC.append(acc)
        BWT.append(bwt)
        FWT.append(fwt)
        Time.append(all_performed_time)
    print('Results of {} repeated runs'.format(Multiple))
    print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)))
    print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)))
    print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)))
    print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)))

    # also saved in the files
    method = 'MRNet'
    fname= method
    fname += datetime.datetime.now().strftime("_%m_%d_%H_%M")  # "%Y_%m_%d_%H_%M_%S"
    fname = os.path.join('FashionMNIST_10_5\Results', fname + '.txt')
    if fname is not None:
        f = open(fname, 'w')
        print('Results of runing {} with {} randomly shuffled task orders'.format(method ,Multiple), file=f)
        print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)), file=f)
        print('Backward: mean {:.4f}, std {:.4f}'.format(np.mean(BWT), np.std(BWT, ddof=1)), file=f)
        print('Forward:  mean {:.4f}, std {:.4f}'.format(np.mean(FWT), np.std(FWT, ddof=1)), file=f)
        print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)), file=f)
        f.close()

