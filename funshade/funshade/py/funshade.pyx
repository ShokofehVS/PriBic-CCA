import numpy as np
cimport numpy as np

from libc.stdint cimport int64_t, int64_t, uint8_t
from libcpp cimport bool


cdef extern from "fss.h" nogil:
    ctypedef int64_t R_t
    const size_t KEY_LEN
    const size_t SEED_LEN

    ## FUNSHADE (batch evaluation)
    void funshade_setup_batch(size_t K, size_t l, R_t theta,
        R_t d_x0[], R_t d_x1[], R_t d_y0[], R_t d_y1[], R_t d_xy0[], R_t d_xy1[],
        R_t r_in_0[],R_t r_in_1[], uint8_t k0[], uint8_t k1[])
    void funshade_share_batch(size_t K, size_t l, const R_t v[], const R_t d_v[],R_t D_v[])
    void funshade_eval_dist_batch(size_t K, size_t l, bint j,
        const R_t r_in_j[], const R_t D_x[], const R_t D_y[],
        const R_t d_xj[], const R_t d_yj[], const R_t d_xyj[], R_t z_hat[])
    void funshade_eval_sign_batch(size_t K, bint j, const uint8_t kj[],
        const R_t z_hat_0[], const R_t z_hat_1[], R_t o_j[])
    R_t funshade_eval_sign_batch_collapse(size_t K, bint j, const uint8_t kj[],
        const R_t z_hat_0[], const R_t z_hat_1[])
    void funshade_setupSS_batch(size_t l, R_t theta,
        R_t a_hat_0[], R_t a_hat_1[], R_t b_hat_0[], R_t b_hat_1[], R_t c_hat_0[], R_t c_hat_1[])
    void funshade_shareSS_batch(size_t l, const R_t v[], const R_t hat_ab[], R_t hat_de[])
    void funshade_node2del_batch(size_t l, bint j, const R_t hat_d[], const R_t hat_e[],
              const R_t hat_a_j[], const R_t hat_b_j[], const R_t hat_c_j[], R_t z_hat_j[])
    void funshade_beaverTriple_square(size_t l, R_t theta,
        R_t a_hat_0[], R_t a_hat_1[], R_t b_hat_0[], R_t b_hat_1[])
    void funshade_shareSquare_batch(size_t l, const R_t v[], const R_t hat_a[], R_t e[])
    void funshade_square_batch(size_t l, bint j, const R_t e[],
              const R_t hat_a_j[], const R_t hat_b_j[], R_t z_hat_j[])


    ## FSS (batch evaulation)
    void SIGN_gen_batch(size_t K, R_t theta, R_t r_in_0[], R_t r_in_1[], uint8_t k0[], uint8_t k1[])
    void SIGN_eval_batch(size_t K, bool b, const uint8_t kb[], const R_t x_hat[], R_t ob[])

# build the corresponding numpy type for R_t (ring type)
cdef R_t tmp = 42
DTYPE = {
    (1, True): np.uint8,  (1, False): np.int8,
    (2, True): np.uint16, (2, False): np.int16, 
    (4, True): np.uint32, (4, False): np.int32, 
    (8, True): np.uint64, (8, False): np.int64, 
}[(sizeof(tmp), (tmp>0)&(~tmp>=0))]

#--------------------------------- FUNSHADE -----------------------------------#
def setup(size_t K, size_t l, R_t theta):
    """Setup for the FunShade protocol.
    
    Generates the beaver triples, input masks and function keys.

    Args:
        K (int): Number of vectors.
        l (int): Number of elements per vector.
        theta (int): Upscaled threshold.
    
    Returns:
        d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1 (np.ndarray): beaver triples for x and y.
        r_in0, r_in1 (np.ndarray): input masks.
        k0, k1 (np.ndarray): function keys.
    """
    cdef np.ndarray[R_t, ndim=1] d_x0  =\
                np.empty((K*l), DTYPE), d_x1  = np.empty((K*l), DTYPE),\
        d_y0  = np.empty((K*l), DTYPE), d_y1  = np.empty((K*l), DTYPE),\
        d_xy0 = np.empty((K*l), DTYPE), d_xy1 = np.empty((K*l), DTYPE),\
        r_in0 = np.empty((K),   DTYPE), r_in1 = np.empty((K),   DTYPE)
        
    cdef np.ndarray[uint8_t, ndim=1] k0 = np.empty((K*KEY_LEN), np.uint8), k1 = np.empty((K*KEY_LEN), np.uint8)
    
    funshade_setup_batch(K, l, theta,
           &d_x0[0], &d_x1[0], &d_y0[0], &d_y1[0], &d_xy0[0], &d_xy1[0], &r_in0[0], &r_in1[0], &k0[0], &k1[0])
    return d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1, r_in0, r_in1, k0, k1

def share(size_t K, size_t l, R_t[::1] v, R_t[::1] d_v):
    """Generate Delta share of a vector v (Pi secret sharing)
    
    Args:
        K (int): Number of vectors.
        l (int): Number of elements per vector.
        v (np.ndarray): Vector to be shared.
        d_v (np.ndarray): Beaver triple input shares for v.
    
    Returns:
        D_v (np.ndarray): Delta share of v.
    """
     # Check size to avoid segfaults
    assert v.shape[0]==d_v.shape[0]==<Py_ssize_t>(K*l),\
        "<Funshade error> Input vector v and delta shares must be of length %d (K*l)".format(K*l)
    cdef np.ndarray[R_t, ndim=1] D_v = np.empty((K*l), DTYPE)
    funshade_share_batch(K, l, &v[0], &d_v[0], &D_v[0])
    return D_v

def eval_dist(size_t K, size_t l, bint j, R_t[::1] r_in_j, R_t[::1] D_x, R_t[::1] D_y, 
              R_t[::1] d_xj, R_t[::1] d_yj, R_t[::1] d_xyj):
    """Compute the distance function (scalar prod.) on the Delta shares of x and y.

    Args:
        K (int): Number of vectors.
        l (int): Number of elements per vector.
        j (bint): Input mask bit.
        r_in_j (np.ndarray): Input mask share.
        D_x (np.ndarray): Delta shares of x.
        D_y (np.ndarray): Delta shares of y.
        d_xj (np.ndarray): Beaver triple input shares for x.
        d_yj (np.ndarray): Beaver triple input shares for y.
        d_xyj (np.ndarray): Beaver triple shares for products xy.
    
    Returns:
        z_hat_j (np.ndarray): shares of the distance function evaluation result.
    """
     # Check array size to avoid segfaults
    assert D_x.shape[0]==D_y.shape[0]==d_xj.shape[0]==d_yj.shape[0]==d_xyj.shape[0]==<Py_ssize_t>(K*l),\
        "<Funshade error> All delta shares must be of length %d (K*l)".format(K*l)
    assert r_in_j.shape[0]==<Py_ssize_t>(K), "<Funshade error> All r_in masks must be of length %d (K)".format(K)
    cdef np.ndarray[R_t, ndim=1] z_hat_j = np.empty((K), DTYPE)
    funshade_eval_dist_batch(K, l, j, &r_in_j[0], &D_x[0], &D_y[0], &d_xj[0], &d_yj[0], &d_xyj[0], &z_hat_j[0])
    return z_hat_j

def eval_sign(size_t K, bint j, uint8_t[::1] k_j, R_t[::1] z_hat_0, R_t[::1] z_hat_1):
    """Compute the sign function (with FSS) given the shares of a public value z_hat.

    Args:
        K (int): Number of vectors.
        j (bint): Input mask bit.
        k_j (np.ndarray): Function key share.
        z_hat_0 (np.ndarray): Shares of z_hat from P0.
        z_hat_1 (np.ndarray): Shares of z_hat from P1.
    
    Returns:
        o_j (np.ndarray): shares of the sign function evaluation result.
    """
    assert z_hat_0.shape[0]==z_hat_1.shape[0]==<Py_ssize_t>(K), \
        "<Funshade error> z_hat shares must be of length %d (K)".format(K)
    assert k_j.shape[0]==<Py_ssize_t>(K*KEY_LEN), \
        "<Funshade error> FSS keys k_j must be of length %d (K*KEY_LEN)".format(K*KEY_LEN)
    cdef np.ndarray[R_t, ndim=1] o_j = np.empty((K), DTYPE)
    funshade_eval_sign_batch(K, j, &k_j[0], &z_hat_0[0], &z_hat_1[0], &o_j[0])
    return o_j

def eval_sign_collapse(size_t K, bint j, uint8_t[::1] k_j, R_t[::1] z_hat_0, R_t[::1] z_hat_1):
    """Compute the sign function (with FSS) given the shares of a public value z_hat.

    Returns a single value (sum of results), using less memory.

    Args:
        K (int): Number of vectors.
        j (bint): Input mask bit.
        k_j (np.ndarray): Function key share.
        z_hat_0 (np.ndarray): Shares of z_hat from P0.
        z_hat_1 (np.ndarray): Shares of z_hat from P1.
    
    Returns:
        o_j (np.ndarray): shares of the sign function evaluation result.
    """
    assert z_hat_0.shape[0]==z_hat_1.shape[0]==<Py_ssize_t>(K), \
        "<Funshade error> z_hat shares must be of length %d (K)".format(K)
    assert k_j.shape[0]==<Py_ssize_t>(K*KEY_LEN), \
        "<Funshade error> FSS keys k_j must be of length %d (K*KEY_LEN)".format(K*KEY_LEN)
    return funshade_eval_sign_batch_collapse(K, j, &k_j[0], &z_hat_0[0], &z_hat_1[0])

#--------------------------------- FSS GATE -----------------------------------#
def FssGenSign(size_t K, R_t theta):
    """FssGenSign generates locally the input masks and the function keys for 2PC sign evaluation in semi-honest setting.
    
    Generates the FSS input masks and FSS keys.

    Args:
        K (int): Number of input values vectors.
        theta (int): Upscaled threshold.
    
    Returns:
        r_in0, r_in1 (np.ndarray): shares of the input masks.
        k0, k1 (np.ndarray): function keys.
    """
    cdef np.ndarray[R_t, ndim=1] r_in0 = np.empty((K), DTYPE), r_in1 = np.empty((K), DTYPE)
    cdef np.ndarray[uint8_t, ndim=1] k0 = np.empty((K*KEY_LEN), np.uint8), k1 = np.empty((K*KEY_LEN), np.uint8)
    SIGN_gen_batch(K, theta, &r_in0[0], &r_in1[0], &k0[0], &k1[0])
    return r_in0, r_in1, k0, k1

def FssEvalSign(size_t K, bool j, uint8_t[::1] k_j, R_t[::1] x_hat):
    """FssEvalSign evaluates the sign function in semi-honest setting.

    Args:
        K (int): Number of input values.
        j (bool): Party index (0 or 1)
        k_j (np.ndarray): Function key shares.
        x_hat (np.ndarray): masked input values.

    Returns:
        o_j (np.ndarray): shares of the sign function evaluation result.
    """
    assert x_hat.shape[0]==<Py_ssize_t>(K), \
        "<FssEvalSign error> x_hat shares must be of length %d (K)".format(K)
    assert k_j.shape[0]==<Py_ssize_t>(K*KEY_LEN), \
        "<FssEvalSign error> FSS keys k_j must be of length %d (K*KEY_LEN)".format(K*KEY_LEN)
    cdef np.ndarray[R_t, ndim=1] o_j = np.empty((K), DTYPE)
    SIGN_eval_batch(K, j, &k_j[0], &x_hat[0], &o_j[0])
    return o_j

def setupextended(size_t K):
    """FssGenSign generates locally the input masks and the function keys for 2PC sign evaluation in semi-honest setting.

    Generates the FSS input masks and FSS keys.

    Args:
        K (int): Number of input values vectors.
        theta (int): Upscaled threshold.

    Returns:
        r_in0, r_in1 (np.ndarray): shares of the input masks.
        k0, k1 (np.ndarray): function keys.
    """
    cdef np.ndarray[R_t, ndim=1] r_in0 = np.empty((K), DTYPE), r_in1 = np.empty((K), DTYPE)
    return r_in0, r_in1

def beaverTriple_ss(size_t l, R_t theta):
    """Setup for the FunBic-CCA protocol.

    Generates the beaver triples.

    Args:
        l (int): Number of elements per vector.
        theta (int): Upscaled threshold.

    Returns:
        a_hat_0, a_hat_1, b_hat_0, b_hat_1, c_hat_0, c_hat_1 (np.ndarray): beaver triples in additive secret sharing.
    """
    cdef np.ndarray[R_t, ndim=1] a_hat_0  =\
                np.empty((l), DTYPE), a_hat_1  = np.empty((l), DTYPE),\
        b_hat_0  = np.empty((l), DTYPE), b_hat_1  = np.empty((l), DTYPE),\
        c_hat_0  = np.empty((l), DTYPE), c_hat_1  = np.empty((l), DTYPE),\

    funshade_setupSS_batch(l, theta,
           &a_hat_0[0], &a_hat_1[0], &b_hat_0[0], &b_hat_1[0], &c_hat_0[0], &c_hat_1[0])

    return a_hat_0, a_hat_1, b_hat_0, b_hat_1, c_hat_0, c_hat_1

def share_ss(size_t l, R_t[::1] v, R_t[::1] hat_ab):
    """Generate d and e shares of a input vector (Arithmetic secret sharing)

    Args:
        l (int): Number of elements per vector.
        v (np.ndarray): Vector to be shared.
        hat_ab (np.ndarray): Beaver triple input shares for v.

    Returns:
        hat_de (np.ndarray): d or e share of v.
    """
     # Check size to avoid segfaults
    cdef np.ndarray[R_t, ndim=1] hat_de = np.empty((l), DTYPE)
    funshade_shareSS_batch(l, &v[0], &hat_ab[0], &hat_de[0])

    return hat_de

def node2del(size_t l, bint j, R_t[::1] hat_d, R_t[::1] hat_e,
              R_t[::1] hat_a_j, R_t[::1] hat_b_j, R_t[::1] hat_c_j):
    """Compute the node deletion on the secret shared node and result after fss gate.

    Args:
        l (int): Number of elements per vector.
        j (bint): Input mask bit.
        hat_d   (np.ndarray): d share of input.
        hat_e   (np.ndarray): e share of input.
        hat_a_j (np.ndarray): Beaver triple input shares.
        hat_b_j (np.ndarray): Beaver triple input shares.
        hat_c_j (np.ndarray): Beaver triple shares for products ab.

    Returns:
        z_hat_j (np.ndarray): shares of the node deletion result.
    """
     # Check array size to avoid segfaults
    cdef np.ndarray[R_t, ndim=1] z_hat_j = np.empty((l), DTYPE)
    funshade_node2del_batch(l, j, &hat_d[0], &hat_e[0], &hat_a_j[0], &hat_b_j[0], &hat_c_j[0], &z_hat_j[0])
    return z_hat_j

def beaverTriple_square(size_t l, R_t theta):
    """Generates the beaver triples for squaring.

    Args:
        l (int): Number of elements per vector.
        theta (int): Upscaled threshold.

    Returns:
        a_hat_0, a_hat_1, b_hat_0, b_hat_1 (np.ndarray): beaver triples for squaring.
    """
    cdef np.ndarray[R_t, ndim=1] a_hat_0  =\
                np.empty((l), DTYPE), a_hat_1  = np.empty((l), DTYPE),\
        b_hat_0  = np.empty((l), DTYPE), b_hat_1  = np.empty((l), DTYPE),\

    funshade_beaverTriple_square(l, theta,
           &a_hat_0[0], &a_hat_1[0], &b_hat_0[0], &b_hat_1[0])

    return a_hat_0, a_hat_1, b_hat_0, b_hat_1


def share_square(size_t l, R_t[::1] v, R_t[::1] hat_a):
    """Generate e share of a input vector (Arithmetic secret sharing)

    Args:
        l (int): Number of elements per vector.
        v (np.ndarray): Vector to be shared.
        hat_a (np.ndarray): Beaver triple input shares for v.

    Returns:
        e (np.ndarray): e share of v.
    """
     # Check size to avoid segfaults
    cdef np.ndarray[R_t, ndim=1] e = np.empty((l), DTYPE)
    funshade_shareSquare_batch(l, &v[0], &hat_a[0], &e[0])

    return e


def square(size_t l, bint j, R_t[::1] e,
              R_t[::1] hat_a_j, R_t[::1] hat_b_j):
    """Compute the squaring.

    Args:
        l (int): Number of elements per vector.
        j (bint): Input mask bit.
        e   (np.ndarray): e share of input.
        hat_a_j (np.ndarray): Beaver triple input shares.
        hat_b_j (np.ndarray): Beaver triple input shares.

    Returns:
        z_hat_j (np.ndarray): shares of the node deletion result.
    """
     # Check array size to avoid segfaults
    cdef np.ndarray[R_t, ndim=1] z_hat_j = np.empty((l), DTYPE)
    funshade_square_batch(l, j, &e[0], &hat_a_j[0], &hat_b_j[0], &z_hat_j[0])

    return z_hat_j