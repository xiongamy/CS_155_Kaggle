# Written in Python 3

from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np
import random
import sys
import math

'''
def circumcenter(x1, y1, x2, y2, x3, y3):
    x = -(-x2*x2*y1 + x3*x3*y1 + x1*x1*y2 - x3*x3*y2 + y1*y1*y2 - y1*y2*y2 - x1*x1*y3
            + x2*x2*y3 - y1*y1*y3 + y2*y2*y3 + y1*y3*y3 - y2*y3*y3) /\
            (2 * (x2*y1 - x3*y1 - x1*y2 + x3*y2 + x1*y3 - x2*y3))
    y = (- x1*x1*x2 + x1*x2*x2 + x1*x1*x3 - x2*x2*x3 - x1*x3*x3 + x2*x3*x3 - x2*y1*y1
            + x3*y1*y1 + x1*y2*y2 - x3*y2*y2 - x1*y3*y3 + x2*y3*y3) /\
            (2 * (-x2*y1 + x3*y1 + x1*y2 - x3*y2 - x1*y3 + x2*y3))
    return [x, y]
 
def obtuse(a, b, c):
    sa = a * a
    sb = b * b
    sc = c * c
    if sa > sb + sc:
        return True
    if sb > sa + sc:
        return True
    if sc > sa + sb:
        return True
    return False

if len(sys.argv) < 2:
    print('Enter number of iterations')
    sys.exit()
'''
    
train_data = np.loadtxt('../data/training_data.txt', delimiter=' ', skiprows=1)
importance = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)

Y = train_data[:, 0]
X = train_data[:, 1:]

'''
num_runs = int(sys.argv[1])

hyperps = np.loadtxt('../data/SVM_hyperps.txt', delimiter=' ', skiprows=1)
while num_runs != 0:
    # decide c_val and dimensions most spread out
    # naive voronoi vertices (all possible circumcenters)
    
    interval_log_cvals = np.max(np.log(hyperps[:, 0])) - np.min(np.log(hyperps[:, 0]))
    interval_dims = np.max(hyperps[:, 1]) - np.min(hyperps[:, 1])
    if interval_log_cvals == 0:
        interval_log_cvals = 1
    if interval_dims == 0:
        interval_dims = 1
    
    centers = []
    circumrs = []
    for i1 in range(len(hyperps)):
        hyperp1 = hyperps[i1]
        x1 = np.log(hyperp1[0]) / interval_log_cvals
        y1 = hyperp1[1] / interval_dims
        for i2 in range(i1 + 1, len(hyperps)):
            hyperp2 = hyperps[i2]
            x2 = np.log(hyperp2[0]) / interval_log_cvals
            y2 = hyperp2[1] / interval_dims
            d12 = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
            for i3 in range(i2 + 1, len(hyperps)):
                hyperp3 = hyperps[i3]
                x3 = np.log(hyperp3[0]) / interval_log_cvals
                y3 = hyperp3[1] / interval_dims
                d13 = math.sqrt((x1 - x3)*(x1 - x3) + (y1 - y3)*(y1 - y3))
                d23 = math.sqrt((x2 - x3)*(x2 - x3) + (y2 - y3)*(y2 - y3))
                if obtuse(d12, d23, d13):
                    continue
                
                c = circumcenter(x1, y1, x2, y2, x3, y3)
                centers.append(c)
                
                dx = c[0] - x1
                dy = c[1] - y1
                circumr = math.sqrt(dx*dx + dy*dy)
                
                dx = c[0] - x2
                dy = c[1] - y2
                c_new = math.sqrt(dx*dx + dy*dy)
                if c_new < circumr:
                    circumr = c_new
                
                dx = c[0] - x3
                dy = c[1] - y3
                c_new = math.sqrt(dx*dx + dy*dy)
                if c_new < circumr:
                    circumr = c_new
                
                circumrs.append(circumr)

    valid_centers = []
    dists = []
    for i in range(len(centers)):
        xc = centers[i][0]
        yc = centers[i][1]
        circumr = circumrs[i]
        is_valid = True
        
        for hyperp in hyperps:
            x = np.log(hyperp[0]) / interval_log_cvals
            y = hyperp[1] / interval_dims
            dx = x - xc
            dy = y - yc
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < circumr:
                is_valid = False
                break
        if is_valid:
            valid_centers.append(centers[i])
            dists.append(circumr)
    
    max_center = valid_centers[0]
    max_dist = dists[0]
    for i in range(1, len(valid_centers)):
        if dists[i] > max_dist:
            max_dist = dists[i]
            max_center = valid_centers[i]
    
    c_val = np.exp(max_center[0] * interval_log_cvals)
    dimensions = int(math.floor(max_center[1] * interval_dims))
    if dimensions < 1:
        dimensions = 1
    if dimensions > len(importance):
        dimensions = len(importance)
    
    
    hyperps = np.concatenate((hyperps, [[c_val, dimensions, 0]]))
    
    
    # start running cross-validation
    print('Running with C value ', c_val , ' and ', dimensions, ' dimensions...')

    indices = [importance[i] for i in range(dimensions)]
    
    total_err = 0.0
    num = 1
    
    X_train = []
    for i in range(len(X)):
        X_train.append([X[i, j] for j in indices])
    X_train = np.array(X_train)
    
    
    kf = KFold(n_splits=5, shuffle=True)
    
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
    
    #if len(hyperps) > 40:
    #    hyperps = hyperps[:-10]
    
    hyperps = sorted(hyperps, key=lambda x : x[2])
    hyperps = np.array(hyperps)
    np.savetxt('../data/SVM_hyperps.txt', hyperps, fmt='%f', delimiter=' ', header='c dimensions error', comments='')
    num_runs -= 1
'''

kf = KFold(n_splits=5, shuffle=True)
    
dimensions = 650
X_train = X[:, importance[:dimensions]]
Y_train = Y

'''
for _ in range(2):
    total_err = 0
    for train_index, test_index in kf.split(X_train):
        train_in, test_in = X_train[train_index], X_train[test_index]
        train_out, test_out = Y_train[train_index], Y_train[test_index]
        clf = svm.SVC(C=32)
        clf.fit(train_in, train_out)
        prd = clf.predict(test_in)
        err = 0.
        for i in range(len(prd)):
            if prd[i] != test_out[i]:
                err += 1
        total_err += err / len(prd)
    print('Error: ', total_err / 5)
'''
    
clf = svm.SVC(C=32)
clf.fit(X_train, Y_train)

test_data = np.loadtxt('../data/test_data.txt', delimiter=' ', skiprows=1)
predicted = clf.predict(test_data[:, importance[:dimensions]])

length = len(predicted)
ids = np.linspace(1, length, num=length)

to_save = np.concatenate((np.transpose([ids]), np.transpose([predicted])), axis=1)
np.savetxt("../submissions/svm_submission4.csv", to_save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')
