import numpy as np
import sys
from random import randrange 


r = 16
b = 64

n_hash1 = b*r
h1 = np.empty((n_hash1, 2))
np.random.seed(seed=0)
for i in range(n_hash1):
    h1[i, 0] = int(np.random.randint(1, 8193))
    h1[i, 1] = int(np.random.randint(0, 8193))


def mapper(key, value):
    global h1
    global b
    global r
    # key: None
    # value: one line of input file
    fullValue = value
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
        yield key_1, fullValue


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that keyt
    if len(values)>1:
        for doc1 in range(len(values)):
            value1 = values[doc1].split()
            doc1_id = int(value1[0][-4:])
            doc1set = set(map(int, value1[1:]))

            for doc2 in range(doc1+1,len(values)):
                value2 = values[doc2].split()
                doc2_id = int(value2[0][-4:])
                doc2set = set(map(int, value2[1:]))
                intersect = doc1set.intersection(doc2set)
                union = doc1set.union(doc2set)
                jaccardi = (1.0*len(intersect))/(1.0*len(union))
                if jaccardi>0.85:
                    yield min(doc1_id, doc2_id), max(doc1_id, doc2_id)








