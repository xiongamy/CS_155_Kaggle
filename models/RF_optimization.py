from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random

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
               fmt='%i', delimiter=',', header='Id,Prediction')

               
if __name__ == '__main__':
    # load data
    train_data = load_data('../data/training_data.txt')
    X_all = train_data[:, 1:]
    y_train = train_data[:, 0]
    importance = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)
    
    hyperps = np.loadtxt('../data/rf_h.txt', delimiter=' ', skiprows=1)
    for i in range(100):
        print(i)
        # randomize hyperparameters
        dim = random.randint(500, 1000)
        crit = random.randint(0, 1)
        m = 0.2 * random.random()
        n = int(np.floor(np.exp(random.uniform(3.5, 7))))
        print(dim, crit, m, n)
        
        X_train = X_all[:, importance[:dim]]

        # train using a random forest
        clf = RandomForestClassifier(n_estimators = n, criterion = ['gini', 'entropy'][crit], max_features = m, oob_score = True)
        clf.fit(X_train, y_train)
        
        print(clf.oob_score_)
        # save results
        hyperps = np.concatenate((hyperps, [[dim, crit, m, n, clf.oob_score_]]))
        hyperps = sorted(hyperps, key=lambda x : -x[4]) # sort by oob score, in descending order
        np.savetxt('../data/rf_h.txt', hyperps,
               fmt='%f', delimiter=' ', header='dim crit m n oob')
    

