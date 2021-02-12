#This files is as slight modification of (a subset of) nmf.py from python's sklearn library

def _update_coordinate_descent(X, W, W_orig, Ht, l1_reg, l2_reg, shuffle):
    from cdnmf_fast import _update_cdnmf_fast
    import numpy as np
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = np.dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.:
        # adds l2_reg only on the diagonal
        HHt.flat[::n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.:
        XHt -= l1_reg

    if shuffle:
        permutation = np.random.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, W_orig, HHt, XHt, permutation)


def factorize(X, W, H, tol=1e-10, max_iter=200, l1_reg_W=0,
                            l1_reg_H=0, l2_reg_W=0, l2_reg_H=0, update_H=True,
                            verbose=0, shuffle=False):
    import numpy as np
    # so W and Ht are both in C order in memory
    W = np.asarray(W, dtype=float, order='C')
    W_orig = W.copy()
    Ht = np.asarray(H.T, dtype=float, order='C')
    Ht_orig = Ht.copy()
    X = np.asarray(X, dtype=float, order='C')


    for n_iter in range(max_iter):
        violation = 0.

        # Update W
        violation += _update_coordinate_descent(X, W, W_orig, Ht, l1_reg_W,
                                                l2_reg_W, shuffle)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, Ht_orig, W, l1_reg_H,
                                                    l2_reg_H, shuffle)

        if n_iter == 0:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter
