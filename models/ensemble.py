import numpy as np

files = ['rf_submission1.csv', 'rf_submission2.csv', 'svm_submission2.csv',
         'svm_submission3.csv', 'svm_submission4.csv']
models = []
         
for f in files:
    models.append(np.loadtxt('../submissions/' + f, skiprows=1, delimiter=','))
N = len(models[0])

# determine which triplet is the most different
max_diff = 0
max_models = []
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        for k in range(j + 1, len(files)):
            print('Files:', files[i], files[j], files[k])
            
            diff = 0
            for r in range(N):
                p1 = models[i][r, 1]
                p2 = models[j][r, 1]
                p3 = models[k][r, 1]
                if p1 != p2 or p2 != p3 or p1 != p3:
                    diff += 1
            print('Number different:', diff)
            
            if diff > max_diff:
                max_diff = diff
                max_models = [i, j, k]
                
print('Max different:', max_diff, 'for models',
      files[max_models[0]], files[max_models[1]], files[max_models[2]])

# generate submission file of ensembled triplet      
m1 = models[max_models[0]]
m2 = models[max_models[1]]
m3 = models[max_models[2]]
            
predictions = []
for i in range(N):
    sum = m1[i, 1] + m2[i, 1] + m3[i, 1]
    if sum > 1:
        predictions.append(1)
    else:
        predictions.append(0)
        
# output predictions in submission format
ids = np.array(range(1, N + 1))
output = np.append(ids.reshape(N, 1), np.array(predictions).reshape(N, 1), 1)
np.savetxt('../submissions/ensemble_submission1.csv', output,
           fmt='%i', delimiter=',', header='Id,Prediction')