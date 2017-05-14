import numpy as np
from data_process import process


# generate some data for training
process_data = process()
X = []
y = []
for i in range(60):
    dim = i + 1
    print ('curr iter: %d' % i)
    #for N in range(10000, 110000, 10000):
    for N in range(50000,150000,10000):
        layer_dims = [np.random.randint(60, 80), 100]
        data = process_data.gen_data(dim, layer_dims, N).astype('float32')
        eigenvlues = process_data.get_eigenvalues(data)
        X.append(eigenvlues)
        y.append(dim)

X = np.array(X)
y = np.array(y)

np.savez('train_data.npz', X=X, y=y)
print('completed data generation')