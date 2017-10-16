import numpy as np
import math

b = 100
r = 2
n_hash = b*r
w = np.empty((n_hash, 8193))
b_h = np.empty((n_hash, 1))
a = 0.01
for i in range(n_hash):
    w_aux = np.random.normal(size=(1, 8193))
    w[i, :] = w_aux / np.linalg.norm(w_aux)
    b_h[i] = np.random.uniform(low=0, high=a)

def mapper(key, value):
    global b,r,w,a,b_h

    value = value.split()
    doc_id = int(value[0][-4:])
    value = value[1:]
    n_hash = w.shape[0]
    x = np.zeros(8193)
    for num in value:
        x[int(num)] = 1
    H1 = np.empty(n_hash)
    for n in range(n_hash):
          H1[n] = math.floor((np.dot(w[n,:].T,x)-b_h[n])/a)
    H1 = H1.reshape((b, r))

    H2 = H1.sum(axis=1).astype(int)
    for i in range(H2.shape[0]):
        key_1 = str(H2[i])+' '+str(i)
        yield key_1, doc_id

def reducer(key, values):

    if len(values)>1:
        yield values[0], values[1]