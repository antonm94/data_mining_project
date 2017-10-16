import numpy as np

r = 16
b = 64
number_of_hash = b*r

m = np.empty((number_of_hash, 2))
np.random.seed(seed=0)
for i in range(number_of_hash):
    m[i, 0] = int(np.random.randint(1, 8193))
    m[i, 1] = int(np.random.randint(0, 8193))


def mapper(key, value):
    global m
    # key: None
    # value: one line of input file
    values = value
    value = value.split()
    value = value[1:]

    m = np.ones(number_of_hash) * np.inf
    for i in range(number_of_hash):
        for num in value:
            m[i] = int(min(m[i], (m[i, 0] * int(num) + m[i, 1]) % 8193))

    m = m.reshape((b, r))
    m = m.sum(axis=1).astype(int)
    for i in range(m.shape[0]):
        key1 = str(m[i])+' '+str(i)
        yield key1, values


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








