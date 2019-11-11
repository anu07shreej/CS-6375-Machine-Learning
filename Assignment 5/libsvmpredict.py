
from svmutil import *
import svmc
import numpy as np
import random
import csv

def normalize(X, low=0, high=1):
    X = np.asanyarray(X)
    minX = np.min(X)
    maxX = np.max(X)
    # Normalize to [0...1]. 
    X = X - minX
    X = X / (maxX - minX)
    # Scale to [low...high].
    X = X * (high-low)
    X = X + low
    return X

def zscore(X):
    X = np.asanyarray(X)
    mean = X.mean()
    std = X.std() 
    X = (X-mean)/std
    return X, mean, std

reader = csv.reader(open('Z:/Sem_2_MCS/MachineLearning/Assignment3/training.csv', 'rb'), delimiter=',')
classes = []
data = []
for row in reader:
    classes.append(int(row[0]))
    data.append([float(num) for num in row[1:]])

data = np.asarray(data)
classes = np.asarray(classes)

# normalize data
means = np.zeros((1,data.shape[1]))
stds = np.zeros((1,data.shape[1]))
for i in xrange(data.shape[1]):
    data[:,i],means[:,i],stds[:,i] = zscore(data[:,i])

# shuffle data
idx = np.argsort([random.random() for i in xrange(len(classes))])
classes = classes[idx]
data = data[idx,:]

# turn into python lists again
classes = classes.tolist()
data = data.tolist()

# formulate as libsvm problem
problem = svm_problem(classes, data)

param=svm_parameter("-q")

# 10-fold cross validation
param.cross_validation=1
param.nr_fold=10

# kernel_type : set type of kernel function (default 2)
#   0 -- linear: u'*v
#   1 -- polynomial: (gamma*u'*v + coef0)^degree
#   2 -- radial basis function: exp(-gamma*|u-v|^2)
#   3 -- sigmoid: tanh(gamma*u'*v + coef0)

#param.kernel_type=LINEAR # 95% (raw), 96% (zscore)
#param.kernel_type=POLY # 96% (raw), 97% (zscore)
param.kernel_type=RBF # 43% (raw), 98% (zscore)
#param.kernel_type=SIGMOID # 39% (raw), 98% (zscore)

# perform validation
accuracy = svm_train(problem,param)
print(accuracy)

# disable cv
param.cross_validation = 0

# training with 80% data
trainIdx = int(0.8*len(classes))
problem = svm_problem(classes[0:trainIdx], data[0:trainIdx])

# build svm_model
model = svm_train(problem,param)

# test with 20% data
# if data was not normalized you would do:
# data = (data-means)/stds
p_lbl, p_acc, p_prob = svm_predict(classes[trainIdx:], data[trainIdx:], model)
print(p_acc)

# perform simple grid search 
#results = []
#for c in range(-3,3):
#   for g in range(-3,3):
#       param.C, param.gamma = 2**c, 2**g
#       acc = svm_train(problem,param)
#       results.append([param.C, param.gamma, acc])

#bestIdx = np.argmax(np.array(results)[:,2])
#print results[bestIdx]