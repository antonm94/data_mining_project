import numpy as np
m = 4000


def z(x, W, b):
    z = np.zeros(m)
    for i in range(m):
        z[i] = np.cos(np.dot(x, W[i]) + b[i])
    return z


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    np.random.seed(0)

    scale = 3.9
    b = np.random.uniform(0.0, 2*np.pi, m)
    w = np.random.normal(0, scale, (m, 400))


    f = lambda x: z(x, w, b)

    res = np.apply_along_axis(f, axis=1, arr=X)

    return X
   # return res


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
    regular = 0.001
    step_size = 0.01
    for j in range(1000):
        for i in range(n):
            x = train[i]
            if not y[i]*np.dot(w,x) > 1:
                grad = -y[i]*x + regular*2*w
                w = w - step_size*grad


    # value: one line of input file
    yield 1, w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    wlenght = len(values[0])
    vlenght = len(values)
    arr = np.concatenate(values).reshape([vlenght, wlenght])#maybe change them
    yield np.asarray(np.asmatrix(arr).mean(0))[0]

