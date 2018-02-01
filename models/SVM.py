# Written in Python 3

from sklearn import svm
from sklearn.model_selection import KFold
import numpy as np

# decide hyperparameter C
# First run:
# 0.01      0.1     1       10      100
# 0.49265   0.2696  0.1665  0.15015 0.1522
# Second run:
# 3         30      300
# 0.1558    0.1495  0.15485
# Third run:
# 5.7       10      18      32      57
# 0.15275   0.15115 0.1513  0.1481  0.15055
# Fourth run:
# 30        32      34      36      38
# 0.1511    0.15    0.15155 0.15035 0.15085
# good C value is ~32

train_data = np.loadtxt('../data/training_data.txt', delimiter=' ', skiprows=1)

Y = train_data[:, 0]
X = train_data[:, 1:]

# kf = KFold(n_splits=5, shuffle=True)
# cs = [30, 32, 34, 36, 38]
# errs = []
# for c_val in cs:
#     total_err = 0.0
#     for train_indices, test_indices in kf.split(X):
#         train_in = []
#         train_out = []
#         for i in train_indices:
#             train_in.append(X[i])
#             train_out.append(Y[i])
#         clf = svm.SVC(C=c_val)
#         clf.fit(train_in, train_out)      
#         test_in = []
#         test_out = []
#         for i in test_indices:
#             test_in.append(X[i])
#             test_out.append(Y[i])        
#         predicted = clf.predict(test_in)
#         err = 0.
#         length = len(test_out)
#         for i in range(length):
#             if predicted[i] != test_out[i]:
#                 err += 1.0
#         err /= length
#         total_err += err
#     errs.append(total_err / 5)
# print(cs)
# print(errs)

test_data = np.loadtxt('../data/test_data.txt', delimiter=' ', skiprows=1)

clf = svm.SVC(C=32)
clf.fit(X, Y)

predicted = clf.predict(test_data)

length = len(predicted)
ids = np.linspace(1, length, num=length)

to_save = np.concatenate((np.transpose([ids]), np.transpose([predicted])), axis=1)
np.savetxt("../submissions/svm_submission2.csv", to_save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')