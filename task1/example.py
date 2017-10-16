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
    # 	-> in this case: key[i] = hashvalue+" "+bandIndex
    # values: all values for that key
    # 	-> in this case: value[i] = full line of document
    if len(values)>1:
        for doc1 in range(0,len(values)):
	    #extract doc id and the shingle set for doc1 and doc2
            docvalue1splitted = values[doc1].split()
            doc1_id = int(docvalue1splitted[0][-4:])
            doc1set = set(map(int, docvalue1splitted[1:]))
            for doc2 in range(doc1+1,len(values)):
                docvalue2splitted = values[doc2].split()
                doc2_id = int(docvalue2splitted[0][-4:])
                doc2set = set(map(int, docvalue2splitted[1:]))
                intersection = doc1set.intersection(doc2set)
                union = doc1set.union(doc2set)
                if len(intersection)>.85*len(union):
                    yield min(doc1_id, doc2_id), max(doc1_id, doc2_id)








