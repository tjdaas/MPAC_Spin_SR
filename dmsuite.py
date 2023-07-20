#importing of important and standard functions, which allows us to run the calculations
import numpy as np
from scipy.linalg import eig

#Some functions that we need that were first implemented/written in Matlab by https://appliedmaths.sun.ac.za/~weideman/research/differ.html and later https://github.com/labrosse/dmsuite/blob/master/dmsuite.py (the new version has removed these functions)

def lagroots(N):
    """
    Compute roots of the Laguerre polynomial of degree N
    Parameters
     ----------
    N   : int
          degree of the Hermite polynomial
    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots
    """
    d0 = np.arange(1, 2*N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d, 1) - np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    return np.real(np.sort(mu))

def poldif(*arg):
    """
    Calculate differentiation matrices on arbitrary nodes.
    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f at arbitrarily specified nodes. The
    differentiation matrices can be computed with unit weights or
    with specified weights.
    Parameters
    ----------
    x       : ndarray
              vector of N distinct nodes
    M       : int
              maximum order of the derivative, 0 < M <= N - 1
    OR (when computing with specified weights)
    x       : ndarray
              vector of N distinct nodes
    alpha   : ndarray
              vector of weight values alpha(x), evaluated at x = x_j.
    B       : int
              matrix of size M x N, where M is the highest derivative required.
              It should contain the quantities B[l,j] = beta_{l,j} =
              l-th derivative of log(alpha(x)), evaluated at x = x_j.
    Returns
    -------
    DM : ndarray
         M x N x N  array of differentiation matrices
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the
    1st, 2nd, ... M-th derivates on arbitrary nodes specified in the array
    x. The nodes must be distinct but are, otherwise, arbitrary. The
    matrices are constructed by differentiating N-th order Lagrange
    interpolating polynomial that passes through the speficied points.
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    .. math::
    f^{(m)}_i = D^{(m)}_{ij}f_j
    This function is based on code by Rex Fuzzle
    https://github.com/RexFuzzle/Python-Library
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    """
    if len(arg) > 3:
        raise Exception('number of arguments is either two OR three')

    if len(arg) == 2:
        # unit weight function : arguments are nodes and derivative order
        x, M = arg[0], arg[1]
        N = np.size(x)
        # assert M<N, "Derivative order cannot be larger or equal to number of points"
        if M >= N:
            raise Exception("Derivative order cannot be larger or equal to number of points")
        alpha = np.ones(N)
        B = np.zeros((M, N))

    elif len(arg) == 3:
        # specified weight function : arguments are nodes, weights and B  matrix
        x, alpha, B = arg[0], arg[1], arg[2]
        N = np.size(x)
        M = B.shape[0]

    I = np.eye(N)                       # identity matrix
    L = np.logical_or(I, np.zeros(N))    # logical identity matrix
    XX = np.transpose(np.array([x,]*N))
    DX = XX-np.transpose(XX)            # DX contains entries x(k)-x(j)
    DX[L] = np.ones(N)                  # put 1's one the main diagonal
    c = alpha*np.prod(DX, 1)             # quantities c(j)
    C = np.transpose(np.array([c,]*N))
    C = C/np.transpose(C)               # matrix with entries c(k)/c(j).
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                # with zeros on the diagonal.
    X = np.transpose(np.copy(Z))        # X is same as Z', but with ...
    Xnew = X

    for i in range(0, N):
        Xnew[i:N-1, i] = X[i+1:N, i]

    X = Xnew[0:N-1, :]                     # ... diagonal entries removed
    Y = np.ones([N-1, N])                # initialize Y and D matrices.
    D = np.eye(N)                      # Y is matrix of cumulative sums

    DM = np.empty((M, N, N))                # differentiation matrices

    for ell in range(1, M+1):
        Y = np.cumsum(np.vstack((B[ell-1, :], ell*(Y[0:N-1, :])*X)), 0) # diags
        D = ell*Z*(C*np.transpose(np.tile(np.diag(D), (N, 1))) - D)    # off-diags
        D[L] = Y[N-1, :]
        DM[ell-1, :, :] = D

    return DM

def lagdif(N, M, b):
    """
    Calculate differentiation matrices using Laguerre collocation.
    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f, at the N Laguerre nodes.
    Parameters
    ----------
    N   : int
          number of grid points
    M   : int
          maximum order of the derivative, 0 < M < N
    b   : float
          scale parameter, real and positive
    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree
         Hermite polynomial, scaled by b
    DM : ndarray
         M x N x N  array of differentiation matrices
    Notes
    -----
    This function returns  M differentiation matrices corresponding to the
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The
    matrices are constructed by differentiating N-th order Hermite
    interpolants.
    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    .. math::
    f^{(m)}_i = D^{(m)}_{ij}f_j
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
    Examples
    --------
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Laguerre
    approximation of the first two derivatives of y = f(x) can be obtained
    as
    >>> N = 32; M = 2; b = 30
    >>> import dmsuite as dm
    >>> x, D = dm.lagdif(N, M, b)      # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.exp(-x)                  # function at Laguerre nodes
    >>> plot(x, y, 'r', x, -D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
    """
    if M >= N - 1:
        raise Exception('number of nodes must be greater than M - 1')

    if M <= 0:
        raise Exception('derivative order must be at least 1')

    # compute Laguerre nodes
    x = 0                               # include origin
    x = np.append(x, lagroots(N-1))     # Laguerre roots
    alpha = np.exp(-x/2)               # Laguerre weights


    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta = np.zeros([M, N])
    d = np.ones(N)

    for ell in range(0, M):
        beta[ell, :] = pow(-0.5, ell+1)*d

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)

    # scale nodes by the factor b
    x = x/b

    for ell in range(M):
        DM[ell, :, :] = pow(b, ell+1)*DM[ell, :, :]

    return x, DM

