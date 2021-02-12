# This is a slight modification of the file with the same name from python's sklearn library
#To compile:
#$ cythonize -i cdnmf_fast.pyx

cimport cython
from libc.math cimport fabs


def _update_cdnmf_fast(double[:, ::1] W, double[:, ::1] W_orig, double[:, :] HHt, double[:, :] XHt,
                       Py_ssize_t[::1] permutation):
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef double grad, pg, hess
    cdef Py_ssize_t i, r, s, t

    with nogil:
        for s in range(n_components):
            t = permutation[s]

            for i in range(n_samples):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in range(n_components):
                    grad += HHt[t, r] * W[i, r]

                # projected gradient
                # If W = W_max, do not increase it (= set grad to 0 if negative)
                # If W = 0, do not decreas it (= set grad to 0 if positive)
                if W[i, t] == W_orig[i,t] and grad < 0. or W[i, t] == 0. and grad > 0.:
                    pg = 0.
                else:
                    pg = grad
                violation += fabs(pg)

                # Hessian
                hess = HHt[t, t]

                if hess != 0:
                    #Update with upper cutoff = W_max and lower cutoff = 0
                    W[i, t] = min(W_orig[i,t], max(W[i, t] - grad / hess, 0.))
                
    return violation
