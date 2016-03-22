import numpy as np

def generate_orthogonal_matrix(N):
    """
    with random permutation of coordinate axes
    :param N: dimension of matrix
    :return: NxN orthogonal matrix contains 0 and 1

    Ref: http://stackoverflow.com/questions/33003341/how-to-randomly-generate-a-nonnegative-orthogonal-matrix-in-numpy
    """
    I = np.eye(N)
    p = np.random.permutation(N)
    return I[p]


