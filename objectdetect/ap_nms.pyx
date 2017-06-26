import numpy as np
cimport numpy as np
import sys

def get_alpha(rho):
    diag = np.arange(rho.shape[0])
    alpha = np.minimum(0., np.diag(rho)[None,:] + np.sum(np.maximum(0., rho), axis=0, keepdims=True) - np.maximum(0., rho) - np.maximum(0., np.diag(rho)[None,:]))
    alpha[diag,diag] = np.sum(np.maximum(0., rho), axis=0) - np.maximum(0., np.diag(rho))
    alpha = np.nan_to_num(alpha, 0.)
    return alpha

def get_rho(np.ndarray s_hat, np.ndarray alpha, np.ndarray phi):
    cdef int i
    cdef int j
    cdef np.ndarray rho = np.zeros_like(s_hat)
    cdef np.ndarray idx
    cdef np.ndarray idx2

    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if i == j:
                idx = np.delete(np.arange(s_hat.shape[1]), i)
                rho[i,j] = s_hat[i,j] - np.max(s_hat[i,idx] + alpha[i,idx]) + np.sum(phi[i,idx])
            else:
                idx = np.delete(np.arange(s_hat.shape[1]), [i,j])
                idx2 = np.delete(np.arange(s_hat.shape[1]), i)
                rho[i,j] = s_hat[i,j] - np.maximum(np.max(s_hat[i,idx] + alpha[i,idx]),
                                            s_hat[i,i] + alpha[i,i] + np.sum(phi[i,idx2]))

    return rho

def get_gamma(np.ndarray s_hat, np.ndarray alpha, np.ndarray phi):
    cdef int i
    cdef int j
    cdef np.ndarray gamma = np.zeros_like(s_hat)
    cdef np.ndarray idx1
    cdef np.ndarray idx2

    for i in range(s_hat.shape[0]):
        for k in range(s_hat.shape[1]):
            idx1 = np.delete(np.arange(s_hat.shape[1]), i)
            idx2 = np.delete(np.arange(s_hat.shape[1]), [i,k])
            gamma[i,k] = s_hat[i,i] + alpha[i,i] - np.max(s_hat[i,idx1] + alpha[i,idx1]) + np.sum(phi[i,idx2])
    return gamma

def get_phi(np.ndarray gamma, np.ndarray r_hat):
    cdef int i
    cdef int k
    cdef np.ndarray phi = np.zeros_like(gamma)
    cdef float term1
    cdef float term2

    for i in range(phi.shape[0]):
        for k in range(phi.shape[0]):
            term1, term2 = np.maximum(0., gamma[k,i] + r_hat[i,k]), np.maximum(0., gamma[k,i])
            if np.isinf(term1) and np.isinf(term2):
                phi[i,k] = 0.
            else:
                phi[i,k] = term1 - term2
    return phi

def get_s_hat(s, wa, wb, wc):
    s_hat = np.zeros((s.shape[0] + 1, s.shape[1] + 1))
    s_hat[:-1,:-1] = wb * s
    diag_idx = np.arange(s.shape[0])
    s_hat[diag_idx, diag_idx] = wa * np.diag(s)
    s_hat[:-1,-1] = -1. * wc
    s_hat[-1,:-1] = -np.inf
    return s_hat

def get_r_hat(r):
    r_hat = np.zeros((r.shape[0] + 1, r.shape[1] + 1))
    r_hat[:-1,:-1] = r
    return r_hat

def get_r(S, c):
    r = np.zeros_like(S)
    idx = np.nonzero(np.diag(c))[0]
    for i in idx:
        for j in idx:
            if i != S.shape[0] and j != S.shape[0]:
                r[i,j] = -(S[i,j] + 1)
    return r

def get_c(alpha, phi, rho, gamma):
    c = np.zeros_like(alpha)
    message = alpha + rho + phi + gamma
    diag_idx = np.arange(c.shape[0])
    
    message = np.nan_to_num(message, 0.)
    for i in range(c.shape[0]):
        idx = np.argmax(message[i,:])
        c[i,idx] = 1.
    return c

def aff_prop(S, iterations=10, damping=0.5, print_every=2, w=[1.,1.,1.,1.]):
    wa, wb, wc, wd = w[0], w[1], w[2], w[3]
    
    s_hat = get_s_hat(S, wa, wb, wc)
    c = np.eye(S.shape[0])
    R = -(S + 0) * wd
    
    diag_idx = np.arange(R.shape[0])
    R[diag_idx, diag_idx] = 0.

    r_hat = get_r_hat(R)
    alpha = np.zeros_like(s_hat)
    gamma = np.zeros_like(s_hat)
    rho = np.zeros_like(s_hat)
    phi = np.zeros_like(s_hat)
    
    for itr in range(iterations):
        rho_old, gamma_old, alpha_old, phi_old = rho, gamma, alpha, phi
        
        # THESE SHOULD BE CORRECT
        alpha = damping * alpha + (1 - damping) * get_alpha(rho)
        phi = damping * phi + (1 - damping) * get_phi(gamma, r_hat)
        rho = damping * rho + (1 - damping) * get_rho(s_hat, alpha, phi)
        gamma = damping * gamma + (1 - damping) * get_gamma(s_hat, alpha, phi)
        
        e1 = np.nan_to_num(np.sum((rho - rho_old)**2), 0.)
        e2 = np.nan_to_num(np.sum((gamma - gamma_old)**2), 0.)
        e3 = np.nan_to_num(np.sum((alpha - alpha_old)**2), 0.)
        e4 = np.nan_to_num(np.sum((phi - phi_old)**2), 0.)
        e = e1 + e2 + e3 + e4
        
    return get_c(alpha, phi, rho, gamma)

