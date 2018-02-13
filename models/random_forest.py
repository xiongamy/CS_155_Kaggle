from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import math

def load_data(filename, skiprows = 1):
    '''
    Function loads data stored in the file filename and returns it as
    a numpy ndarray.
    
    Inputs:
        filename: given as a string.
        
    Outputs:
        Data contained in the file, returned as a numpy ndarray
    '''
    return np.loadtxt(filename, skiprows=skiprows, delimiter=' ')
    
def save_predictions(filename, y_predict):
    # save to file in submission format
    N = len(y_predict)
    ids = np.array(range(1, N + 1))
    output = np.append(ids.reshape(N, 1), y_predict.reshape(N, 1), 1)
    np.savetxt(filename, output,
               fmt='%i', delimiter=',', header='Id,Prediction', comments='')

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

           
if __name__ == '__main__':
    # load data
    train_data = load_data('../data/training_data.txt')
    X_all = train_data[:, 1:]
    y_train = train_data[:, 0]
    importance = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)
    
    dim = 800
    n = 1000
    m = 0.0125
    X_train = X_all[:, importance[:dim]]
    
    # train using a random forest
    print('Training')
    clf = RandomForestClassifier(n_estimators = n, criterion = 'entropy', max_features = m, oob_score = True)
    clf.fit(X_train, y_train)
        
    print(clf.oob_score_)
    
    # predict on test data and save
    X_test = load_data('../data/test_data.txt')[:, importance[:dim]]
    y_predict = clf.predict(X_test)
    save_predictions("../submissions/rf_submission2.csv", y_predict)
    

