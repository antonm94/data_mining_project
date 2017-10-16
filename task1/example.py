import numpy as np
import sys
from random import randrange 


b = 16
r = 4
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
    global docIDtoDocSet
    # key: None
    # value: one line of input file
    fullValue = value
    value = value.split()
    doc_id = int(value[0][-4:])
    value = value[1:]

    #put the document into the dictionary
    #print("doc_id and set")
    #print(doc_id)
    #print(set(map(int, value)))
    #docIDtoDocSet[doc_id]=set(map(int, value))
    #print(len(docIDtoDocSet))
    
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
    global docIDtoDocSet
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # check if jaccard distances are right
    #print(key)
    #print(values)
    if len(values)>1:
        for doc1 in range(len(values)):
            value1 = values[doc1].split()
            doc1_id = int(value1[0][-4:])
            doc1set = set(map(int, value1[1:]))
            #print(value1)
            #print(doc1_id)
            #print(doc1set)
            for doc2 in range(doc1+1,len(values)):
                value2 = values[doc2].split()
                doc2_id = int(value2[0][-4:])
                doc2set = set(map(int, value2[1:]))
                #print(values[doc1])
                intersect = doc1set.intersection(doc2set)
                union = doc1set.union(doc2set)
                if(len(intersect)>0.85*len(union)):
                    print(doc1_id)
                    print(doc2_id)
                    yield doc1_id, doc2_id
                    if((int(doc1_id)==219 or int(doc2_id)==219)and(int(doc1_id)==144 or int(doc2_id)==144)): 
                        print(doc1_id)
                        print(doc1set)
                        print(doc2_id)
                        print(doc2set)
                        print("intersect")
                        print(intersect)
                        print("union")
                        print(union)
                        print(1.0*len(intersect)/1.0*len(union))







