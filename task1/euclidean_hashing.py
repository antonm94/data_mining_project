import numpy as np
import math


def mapper(key, string, w,b,a, dicc):
    s = string.split()
    doc_id = int(s[0][-4:])
    s = s[1:]
    n_hash = w.shape[0]
    n_band = 20
    r = 50
    x = np.zeros(8193)
    H1 = np.empty(n_hash)
    for num in s:
        x[int(num)] = 1

    for n in range(n_hash):
          H1[n] = math.floor((np.dot(w[n,:],x)-b[n])/a)


    H1 = H1.reshape((n_band, r))
    H2 = H1.sum(axis=1).astype(int)
    print(doc_id)
    print(H2)
    for i in range(H2.shape[0]):

        k = str(i) + ' ' + str(H2[i])
        if k not in dicc:
            dicc[k] = [doc_id, ]
        else:
            dicc[k].append(doc_id)
    return dicc


def reducer(dicc):
    for e in dicc:
        if len(dicc[e]) > 1:
            print(e)
            print(dicc[e])
    return


def main():
    dicc = {}
    n_hash = 1000
    w = np.empty((n_hash,8193))
    b = np.empty((n_hash,1))
    a = 0.000005
    for i in range(n_hash):
        w_aux = np.random.normal(size=(1,8193))
        w[i,:] = w_aux/np.linalg.norm(w_aux)
        b[i] = np.random.uniform(low=0,high=a)

    with open(filename) as file:
        for line in file:
            dicc = mapper(None, line,w,b,a, dicc)

    reducer(dicc)


if __name__ == '__main__':
    filename = 'data/handout_shingles.txt'
    main()