import numpy as np
from scipy.spatial.distance import cdist
#import logging


#logger = logging.getLogger(__name__)

#N_CORESETS = 2 # Needs to be a power of 2 for correct tree
CORESET_SIZE = 1700
N_CLUSTERS = 200 # Specified by problem description


def init_centers_d2_sampling(X, n_clusters):
    """ D^2 sampling algorithm to find a proper cluster centers given a dataset of points """
    # Extract some values
    N, d = X.shape

    # Centers container
    centers = np.empty((n_clusters, d), dtype=X.dtype)
    # Squared distance of each point to its closest center
    dist = np.empty((N, n_clusters), dtype='float')

    indexes = np.arange(N)
    p = np.ones((N))
    p /= p.sum()

    for i in range(n_clusters):
        # Sample following the given probability distribution
        idx = np.random.choice(indexes, p=p)
        # And store the sampled point into the centers container
        centers[i] = X[idx]

        # Squared distance of each point to its closest center
        dist[:,i] = euclidean_distance(X, centers[i])
        min_dist = dist[:,:i+1].min(axis=1) # Distance to closest center

        # Compute the probability distribution normalizing the squared distance
        p = min_dist / min_dist.sum()

    return centers


def sample_coreset(X,n_clusters):
    """
    :param X: data set to be compressed
    :param n_clusters: dimension of the reduced dataset
    :return: new reduced dataset with corresponding weight per point
    """
    n_b,d = X.shape
    coreset = np.empty((n_clusters,d),dtype=float)
    weights = np.empty((n_clusters,1),dtype=float)

    # Step 1: D^2 Sampling (rough approximation)
    B = init_centers_d2_sampling(X,n_clusters)

    # Step 2: Sample coreset using importance sampling
    # Construct vector q(x) to sample
    dist = cdist(X,B)
    squared_dist = np.min(dist,axis=1)**2
    cluster_membership = np.argmin(dist,axis=1)
    c_phi = np.mean(squared_dist)
    alpha = 5.5
    # First term of q(x)
    first_term = alpha*squared_dist/c_phi
    # Second term of q(x) (might be slow due to for loop)
    B_i = []
    B_i_array = np.zeros((n_b,n_b),dtype=int)
    for i in range(n_clusters):
        B_i.append(np.where(cluster_membership==i))
    for i in range(n_b):
        B_i_array[i,B_i[cluster_membership[i]]] = 1
    second_term = 2*alpha/c_phi*np.divide(np.dot(B_i_array,squared_dist),np.sum(B_i_array,axis=1))
    # Third term of q(x)
    third_term = 4*n_b/np.sum(B_i_array,axis=1)
    q = first_term + second_term + third_term
    q = q/q.sum()

    # Sample from q(x)
    indexes = np.arange(n_b)
    for i in range(n_clusters):
        idx = np.random.choice(indexes, p=q)
        coreset[i,:] = X[idx,:]
        weights[i,0] = 1/q[idx]

    # Stack cluster centers with weights
    C = np.hstack((weights,coreset))

    return C

def euclidean_distance(X, Y):
    """ Compute the euclidean distance pair-wise between column vectors of X and Y """
    if len(Y.shape) == 1:
        return np.sum((X-Y)**2, axis=1)

    assert X.shape[1] == Y.shape[1], 'Last dimension do not match'

    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        result[i,:] = euclidean_distance(Y, X[i,:])
    return result


def kmeans_coresets(X, w, n_clusters=8, n_init=10, max_iter=300, tol=.0001):
    """ Fit the K-Means cluster algorithm with the coresets represented by the points `X` and
    weights `w` """

    assert X.shape[0] == w.shape[0], \
        "X and w must have the same number of samples. {} != {}".format(X.shape[0], w.shape[0])

    best_centers, best_inertia, best_labels = None, None, None

    n_samples = X.shape[0]

    for i in range(n_init):

        # Initialize the centers using the k-means++ algorithm
        centers = init_centers_d2_sampling(X, n_clusters)

        it = 0
        prev_L = 0
        while it < max_iter:

            L = 0
            # Assign to each point the index of the closest center
            labels = np.zeros(n_samples, dtype='int')
            for j in range(n_samples):
                d_2 = np.sum((centers-X[j,:])**2, axis=1)
                labels[j] = np.argmin(d_2)
                L += w[i,0] * d_2[labels[j]]
            L /= w.sum()

            # Update
            for l in range(n_clusters):
                if np.sum(labels==l) == 0:
                    continue
                P = X[labels==l,:]
                pw = w[labels==l,:]
                centers[l] = np.sum(pw * P, axis=0) / pw.sum()

            # Check convergence
            if abs(prev_L - L) < tol:
                break
            prev_L = L

            it += 1

       # logger.info('Finished with {} iterations!'.format(it))


        # Compute intertia and update the best parameters
        inertia = L
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return best_centers

def mapper(key, value):
    """
    # key: None
    # value: one line of input file
    For each batch of data:
        1. Extract 128 coresets.
        2. Union and compression until just 1 coreset.
        3. Yield coreset
    """
    # Get dimensions
    X = value
    n_b = X.shape[0]
    d = X.shape[1]


    """
    # Create first row of coresets
    coresets = {}
    n_levels = int(math.log(N_CORESETS,2))
    for i in range(1,n_levels+1):
        coresets[i] = []
        for j in range((N_CORESETS)/(2**(i-1))):
            if i == 1:
                # Sample coresets
                coresets[i].append(sample_coreset(X,CORESET_SIZE))
            else:
                c1 = coresets[i-1].pop()[:,1:]
                c2 = coresets[i-1].pop()[:,1:]
                c = np.vstack((c1,c2))
                coresets[i].append(sample_coreset(c,CORESET_SIZE))

    c1 = coresets[n_levels].pop()[:, 1:]
    c2 = coresets[n_levels].pop()[:, 1:]
    c = np.vstack((c1, c2))
    """

    C = sample_coreset(X,CORESET_SIZE)



    # Just sample some coresets and yield them together with the weight
    # Train it in reducing phase.

    #logger.info('Finished mapper')
    yield 'w', C  # this is how you yield a key, value pair

def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    assert key == 'w', 'Key is has not the correct value'

   # logger.info('Starting reducer')

    w, X = values[:,0].reshape(-1, 1), values[:,1:].reshape(-1,250)
    cluster_centers = kmeans_coresets(X, w, N_CLUSTERS, 1)

   # logger.info('Finished reducer')
    yield cluster_centers
