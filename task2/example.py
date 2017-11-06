import numpy as np

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.


    return X


def mapper(key, value):
    # key: None
    n = len(value)
    k = 400
    train = np.zeros((n, k))
    y = np.zeros(n)
    np.random.shuffle(value)

    for i in range(n):
        v = map(float, value[i].split())
        y[i] = v[0]
        train[i] = v[1:]
    train = transform(train)
    [n, k] = train.shape
    w = np.zeros(k)



    # value: one line of input file
    yield "key", "value"  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    # wlenght = len(values[0])
    # vlenght = len(values)
    # arr = np.concatenate(values).reshape([vlenght, wlenght])#maybe change them
    # yield np.asarray(np.asmatrix(arr).mean(0))
    yield np.random.rand(400)