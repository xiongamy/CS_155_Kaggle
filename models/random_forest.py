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
    
    '''hyperps = np.loadtxt('../data/rf_h.txt', delimiter=' ', skiprows=1)
    for i in range(100):
        
        interval_m = np.max(hyperps[:, 0] / np.sqrt(hyperps[:, 1])) - np.min(hyperps[:, 0] / np.sqrt(hyperps[:, 1]))
        interval_dims = np.max(hyperps[:, 1]) - np.min(hyperps[:, 1])
        if interval_m == 0:
            interval_m = 1
        if interval_dims == 0:
            interval_dims = 1
    
        centers = []
        circumrs = []
        for i1 in range(len(hyperps)):
            hyperp1 = hyperps[i1]
            x1 = hyperp1[0] / interval_m
            y1 = hyperp1[1] / interval_dims
            for i2 in range(i1 + 1, len(hyperps)):
                hyperp2 = hyperps[i2]
                x2 = hyperp2[0] / interval_m
                y2 = hyperp2[1] / interval_dims
                d12 = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
                for i3 in range(i2 + 1, len(hyperps)):
                    hyperp3 = hyperps[i3]
                    x3 = hyperp3[0] / interval_m
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
                x = hyperp[0] / interval_m
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
        
        if len(valid_centers) > 0:
            print('Generated')
            max_center = valid_centers[0]
            max_dist = dists[0]
            for i in range(1, len(valid_centers)):
                if dists[i] > max_dist:
                    max_dist = dists[i]
                    max_center = valid_centers[i]
            
            m = max_center[0] * interval_m * math.sqrt(max_center[1])
            dim = int(math.floor(max_center[1] * interval_dims))
            if dimensions < 1:
                dimensions = 1
            if dimensions > len(importance):
                dimensions = len(importance)
        else:
            print('Random')
            dim = random.randint(700, 1000)
            m = random.randint(1, int(math.floor(2*math.sqrt(dim))))
        
        
        
        
        n = 500
        
        print('Running with m value ', m, ' and ', dim, ' dimensions.')
        
        X_train = X_all[:, importance[:dim]]
        
        # train using a random forest
        clf = RandomForestClassifier(n_estimators = n, criterion = 'entropy', max_features = m, oob_score = True)
        clf.fit(X_train, y_train)
        
        print(clf.oob_score_)
        
        if len(hyperps) > 50:
            hyperps = hyperps[:-10]
        
        # save results
        hyperps = np.concatenate((hyperps, [[m, dim, clf.oob_score_]]))
        hyperps = sorted(hyperps, key=lambda x : x[2], reverse=True) # sort by oob score, in descending 
        
        hyperps = np.array(hyperps)
        np.savetxt('../data/rf_h.txt', hyperps,
               fmt='%f', delimiter=' ', header='dim m oob', comments='')'''
    
    dim = 800
    n = 1000
    m = 10
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
    

