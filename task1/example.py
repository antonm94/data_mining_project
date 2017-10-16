import numpy as np
import sys
from random import randrange

n_hash1 = 1100
b = 55
r = 20
h1 = np.empty((n_hash1, 2))
for i in range(n_hash1):
    h1[i, 0] = int(np.random.randint(1, 8193))
    h1[i, 1] = int(np.random.randint(0, 8193))



def mapper(key, value):
    global h1
    global b
    global r
    # key: None
    # value: one line of input file

    value = value.split()
    doc_id = int(value[0][-4:])
    value = value[1:]
    n_hash1 = h1.shape[0]
    H1 = np.ones(n_hash1) * np.inf
    for i in range(n_hash1):
        for num in value:
            H1[i] = int(min(H1[i], ((h1[i, 0] * int(num) + h1[i, 1])) % 8193))

    H1 = H1.reshape((b, r))

    H2 = H1.sum(axis=1).astype(int)
    for i in range(H2.shape[0]):
        key_1 = str(H2[i])+' '+str(i)
        yield key_1, doc_id

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    if len(values)>1:
        yield values[0], values[1]





