import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def l2_dist(p1, p2):
    s = 0
    for i in range(len(p1)):
        s += (p1[i] - p2[i]) * (p1[i] - p2[i])
    return math.sqrt(s)

def uniform_from_simplex(points):
    ps = np.array(points)
    s = ps[0]
    for i in range(1, len(ps)):
        r = random.random()
        prop = r ** (1. / i)
        s = s * prop + ps[i] * (1 - prop)
    return s

def random_from_convex(points, dim):
    # get random simplex
    simplex = []
    indices = np.random.permutation(len(points))
    # 6 random points (1 more than dimension)
    for i in range(dim+1):
        simplex.append(points[indices[i]])
    return uniform_from_simplex(simplex)

def farthest_in_convex(points):
    n, d = points.shape
    norm_points = np.zeros((n, 1))
    stds = []
    for i in range(d):
        col = points[:, i:i+1]
        std = np.std(col)
        stds.append(std)
        if std != 0:
            norm_points = np.concatenate((norm_points, col / std), axis=1)
        else:
            norm_points = np.concatenate((norm_points, col), axis=1)
    norm_points = norm_points[:, 1:]

    max_dist = -float('inf')
    max_point = []
    for i in range(1000):
        r_point = random_from_convex(norm_points, d)
        dist_to_closest = float('inf')
        for p in norm_points:
            dist = l2_dist(p, r_point)
            if dist < dist_to_closest:
                dist_to_closest = dist
        if dist_to_closest > max_dist:
            max_dist = dist_to_closest
            max_point = r_point

    for i in range(d):
        if stds[i] != 0:
            max_point[i] *= stds[i]
    return max_point



# load data into Keras format
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

train_data = np.loadtxt('../data/training_data.txt', delimiter=' ', skiprows=1)
imp = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)
test_data = np.loadtxt('../data/test_data.txt', delimiter=' ', skiprows=1)

gen = 0

#while True:
grand = np.loadtxt('../data/nnha.txt', delimiter=' ', skiprows=1)
gen += 1
'''
h = random.randint(100, 1000)
n1 = random.randint(1, 15)
d = random.uniform(0.0, 1.0)
n2 = random.randint(1, 100)
'''

# Evolutionary training of hyperparams

#r_point = farthest_in_convex(grand[:200, :-1])
r_point = grand[0]

h = int(np.round(r_point[0]))
n1 = int(np.round(r_point[1]))
d = r_point[2]
n2 = int(np.round(r_point[3]))

'''
h = int(grand[c, 0])
hr = random.randint(max(1 - h, -10), min(1000 - h, 10))
h += hr

n1 = int(grand[c, 1])
n1r = random.randint(max(1 - n1, -2), 2)
n1 += n1r

d = grand[c, 2]
dr = random.uniform(max(-d, -0.03), min(1 - d, 0.03))
d += dr

n2 = int(grand[c, 3])
n2r = random.randint(max(1 - n2, -5), 5)
n2 += n2r

e = int(grand[c, 4])
er = random.randint(max(1 - e, -2), 2)
e += er
'''

y_train = train_data[:, 0]
x_temp = train_data[:, 1:]

x_train = np.zeros((len(x_temp), h))
count = 0
while count < h:
    for i in range(len(x_temp)):
        x_train[i, count] = x_temp[i, count]
    count += 1

x_temp = test_data[:, 1:]
x_test = np.zeros((len(x_temp), h))
count = 0
while count < h:
    for i in range(len(x_temp)):
        x_test[i, count] = x_temp[i, count]
    count += 1

# we'll need to one-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)


acc = 0
#kf = KFold(n_splits=5, shuffle=True)
#for train_index, test_index in kf.split(x_train):
#    x_tr, x_te = x_train[train_index], x_train[test_index]
#    y_tr, y_te = y_train[train_index], y_train[test_index]
model = Sequential()
model.add(Dense(n1, input_shape=(h,)))
model.add(Activation('relu'))
model.add(Dropout(d))
model.add(Dense(n2))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

# For a multi-class classification problem
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
'''
x_predict = model.predict_classes(x=x_te)
e_out = 0.0
for i in range(len(x_predict)):
    if (x_predict[i] != y_te[i, 1]):
        e_out += 1
acc += e_out / len(x_predict)
'''
acc /= 5
#print(acc)
'''
print("Iteration: ", len(grand) - 199)
print("Top: ", grand[0])
print("Generated: ", [h, n1, d, n2, acc])
print("200: ", grand[199], "\n")

grand = np.concatenate((grand, [[h, n1, d, n2, acc]]))
grand = sorted(grand, key=lambda x: x[4])

np.savetxt("../data/nnha.txt", grand, fmt='%f', delimiter=' ', header='h n1 d n2 err', comments='')
'''


y_test = model.predict_classes(x=x_test)

ids = np.linspace(1, len(y_test), num=len(y_test))

save = np.concatenate((np.transpose([ids]), np.transpose([y_test])), axis=1)

np.savetxt("../submissions/nn_submission2.csv", save, fmt='%i', delimiter=',', header='Id,Prediction', comments='')
