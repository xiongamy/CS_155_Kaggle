from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

if __name__ == '__main__':
    # load data
    train_data = load_data('../data/training_data.txt')
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = load_data('../data/test_data.txt')

    # train using a random forest
    clf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy')
    clf.fit(X_train, y_train)
    
    # predict on test data
    y_predict = clf.predict(X_test)
    
    # save to file in submission format
    N = len(y_predict)
    ids = np.array(range(1, N + 1))
    output = np.append(ids.reshape(N, 1), y_predict.reshape(N, 1), 1)
    np.savetxt("../submissions/rf_submission1.txt", output,
               fmt='%i', delimiter=',', header='Id,Prediction')
