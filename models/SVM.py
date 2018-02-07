# Written in Python 3

from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
import random
import sys
import math

def circumcenter(x1, y1, x2, y2, x3, y3):
    x = -((-x2*x2*y1 + x3*x3*y1 + x1*x1*y2 - x3*x3*y2 + y1*y1*y2 - y1*y2*y2 - x1*x1*y3
            + x2*x2*y3 - y1*y1*y3 + y2*y2*y3 + y1*y3*y3 - y2*y3*y3) /\
            (2 * (x2*y1 - x3*y1 - x1*y2 + x3*y2 + x1*y3 - x2*y3)))
    y = (- x1*x1*x2 + x1*x2*x2 + x1*x1*x3 - x2*x2*x3 - x1*x3*x3 + x2*x3*x3 - x2*y1*y1
            + x3*y1*y1 + x1*y2*y2 - x3*y2*y2 - x1*y3*y3 + x2*y3*y3) /\
            (2 * (-x2*y1 + x3*y1 + x1*y2 - x3*y2 - x1*y3 + x2*y3))
    return [x, y]

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

first = True
hyperps = np.loadtxt('../data/SVM_hyperps.txt', delimiter=' ', skiprows=1)
while num_runs != 0:
    # decide c_val and dimensions most spread out
    # naive voronoi vertices (all possible circumcenters)
    centers = []
    for hyperp1 in hyperps:
        x1 = np.log(hyperp1[0]) / 7.
        y1 = hyperp1[1] / 1000.
        for hyperp2 in hyperps:
            x2 = np.log(hyperp2[0]) / 7.
            y2 = hyperp2[1] / 1000.
            d12 = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
            if d12 == 0:
                continue
            for hyperp3 in hyperps:
                x3 = np.log(hyperp3[0]) / 7.
                y3 = hyperp3[1] / 1000.
                d13 = math.sqrt((x1 - x3)*(x1 - x3) + (y1 - y3)*(y1 - y3))
                d23 = math.sqrt((x2 - x3)*(x2 - x3) + (y2 - y3)*(y2 - y3))
                if d23 == 0 or d13 == 0:
                    continue
                if d12 + d23 <= d13:
                    continue
                if d12 + d13 <= d23:
                    continue
                if d13 + d23 <= d12:
                    continue
                c = circumcenter(x1, y1, x2, y2, x3, y3)
                if c[0] * 7 >= -1 and c[0] * 7 <= 6 and c[1] >= 0 and c[1] <= 1:
                    centers.append(c)
    
    dists = []
    for center in centers:
        xc = center[0]
        yc = center[1]
        min_dist = float('inf')
        for hyperp in hyperps:
            x = np.log(hyperp[0]) / 7.
            y = hyperp[1] / 1000.
            dx = x - xc
            dy = y - yc
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
        dists.append(min_dist)
    
    max_center = centers[0]
    max_dist = dists[0]
    for i in range(1, len(centers)):
        if dists[i] > max_dist:
            max_dist = dists[i]
            max_center = centers[i]
    
    print(max_center)
    c_val = np.exp(max_center[0] * 7)
    dimensions = int(math.floor(max_center[1] * 1000))
    if dimensions < 0:
        dimensions = 0
    if dimensions > len(importance):
        dimensions = len(importance)
    
    # store hyperparameters
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
    num_runs -= 1
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