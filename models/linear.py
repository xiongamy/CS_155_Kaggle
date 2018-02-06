from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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

    n, d = X_train.shape
    
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    weights = np.array(linreg.coef_)
    
    indices = list(range(d))
    table = {}
    for i in indices:
        table[str(i)] = weights[i]
    
    
    importance = sorted(indices, key=lambda x : abs(table[str(x)]), reverse=True)

    np.savetxt("../data/order.txt", [importance], fmt='%i', delimiter=' ', comments='')