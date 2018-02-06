# Written in Python 3

from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
import random
import sys

if len(sys.argv) < 2:
    print('Enter number of iterations')
    sys.exit()

train_data = np.loadtxt('../data/training_data.txt', delimiter=' ', skiprows=1)
importance = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)

Y = train_data[:, 0]
X = train_data[:, 1:]

kf = KFold(n_splits=5, shuffle=True, random_state=0)
min_log_cval = -1
max_log_cval = 6
num_runs = int(sys.argv[1])

hyperps = np.loadtxt('../data/SVM_hyperps.txt', delimiter=' ', skiprows=1)
for _ in range(num_runs):
    # choose c_val and dimensions
    c_val = np.exp(random.uniform(min_log_cval, max_log_cval))
    dimensions = random.randint(1, len(importance))
    for hyperp in hyperps:
        if random.random() < 0.1:
            if random.randint(0, 1) == 0:
                c_val = hyperp[0]
            else:
                dimensions = int(hyperp[1])
            break

    hyperps = np.concatenate((hyperps, [[c_val, dimensions, 0]]))
    
    print('Running with C value ', c_val , ' and ', dimensions, ' dimensions...')
    
    indices = [importance[i] for i in range(dimensions)]
    
    total_err = 0.0
    num = 1
    
    X_train = []
    for i in range(len(X)):
        X_train.append([X[i, j] for j in indices])
    X_train = np.array(X_train)
    
    
    for train_indices, test_indices in kf.split(X_train):
        train_in = []
        train_out = []
        for i in train_indices:
            train_in.append(X_train[i])
            train_out.append(Y[i])
        clf = svm.SVC(C=c_val)
        
        print('    Running ', num)
        
        clf.fit(train_in, train_out)      
        test_in = []
        test_out = []
        for i in test_indices:
            test_in.append(X_train[i])
            test_out.append(Y[i])        
        predicted = clf.predict(test_in)
        err = 0.
        length = len(test_out)
        for i in range(length):
            if predicted[i] != test_out[i]:
                err += 1.0
        err /= length
        print('    Test error: ', str(err))
        total_err += err
        num += 1
    hyperps[-1, -1] = total_err / 5
    hyperps = sorted(hyperps, key=lambda x : x[2])
    np.savetxt('../data/SVM_hyperps.txt', np.array(hyperps), fmt='%f', delimiter=' ', header='c dimensions error', comments='')
'''
test_data = np.loadtxt('../data/test_data.txt', delimiter=' ', skiprows=1)

clf = svm.SVC(C=32)
clf.fit(X, Y)

predicted = clf.predict(test_data)

length = len(predicted)
ids = np.linspace(1, length, num=length)

to_save = np.concatenate((np.transpose([ids]), np.transpose([predicted])), axis=1)
np.savetxt("../submissions/svm_submission2.csv", to_save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')
'''