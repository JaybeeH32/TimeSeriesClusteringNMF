import numpy as np
import cvxpy as cp
from math import sqrt
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, sigmoid_kernel

# NMF related functions
def sparse_nmf(V, d, beta, eta, max_iters=100):
    '''
    Implementation according to https://faculty.cc.gatech.edu/~hpark/papers/GT-CSE-08-01.pdf
    
    Beta controls L1 norm of H
    Eta controls norm of W
    '''
    n = len(V)
    residual = np.zeros(max_iters)
    H = np.random.random(size=(d, n))
    W = np.random.random(size=(n, d))
    modif_V = np.zeros(shape=(n+1, n))
    modif_V[:n, :n] = V
    
    modif_Vt = np.zeros(shape=(n+d, n))
    modif_Vt[:n, :n] = V.T
    
    for iter_num in range(1, 1+max_iters):
        if iter_num % 2 == 1:
            H = cp.Variable(shape=(d, n))
            constraint = [H >= 0]
            modif_W = sqrt(beta) * np.ones(shape=(n+1, d))
            modif_W[:n, :d] = W
            
            obj = cp.Minimize(cp.norm(modif_V - modif_W@H, 'fro'))
            prob = cp.Problem(obj, constraint)
            prob.solve(solver=cp.SCS, max_iters=10000)

            if prob.status != cp.OPTIMAL:
                raise Exception("Solver did not converge!")
            
            H = H.value
        # For even iterations, treat H constant, optimize over W.
        else:
            W = cp.Variable(shape=(n, d))
            constraint = [W >= 0]
            modif_Ht = np.zeros(shape=(n+d, d))
            modif_Ht[:n, :d] = H.T
            modif_Ht[n:, :d] = sqrt(eta) * np.eye(d, d)
            
            obj = cp.Minimize(cp.norm(modif_Vt - modif_Ht@W.T, 'fro'))
            prob = cp.Problem(obj, constraint)
            prob.solve(solver=cp.SCS, max_iters=10000)

            if prob.status != cp.OPTIMAL:
                raise Exception("Solver did not converge!")
            
            W = W.value
            
        #print(f'Iteration {iter_num}, residual norm {prob.value}')
        residual[iter_num-1] = prob.value
    print(f'Iteration {iter_num}, residual norm {prob.value}')
    return W, H, residual

def nmf(V, d, max_iters=100):
    n = len(V)
    W = np.random.random(size=(n, d))
    H = np.random.random(size=(d, n))
    
    residual = np.zeros(max_iters)
    for iter_num in range(1, 1+max_iters):
        if iter_num % 2 == 1:
            H = cp.Variable(shape=(d, n))
            constraint = [H >= 0]
        # For even iterations, treat X constant, optimize over Y.
        else:
            W = cp.Variable(shape=(n, d))
            constraint = [W >= 0]

        obj = cp.Minimize(cp.norm(V - W@H, 'fro'))
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, max_iters=10000)

        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        #print(f'Iteration {iter_num}, residual norm {prob.value}')
        residual[iter_num-1] = prob.value

        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            H = H.value
        else:
            W = W.value
    print(f'Iteration {iter_num}, residual norm {prob.value}')
    return W, H, residual

def nmf_sklearn(A, d):
    nm = NMF(n_components=d)
    w = nm.fit_transform(A)
    h = nm.components_
    return w, h, np.linalg.norm(A - w@h)

def semi_nmf(V, d, max_iters=100):
    """
    Algorithm taken from "Convex and Semi-Nonnegative Matrix Factorizations", Ding, Li, Jordan

    """
    n = np.shape(V)[0]
    # initialization
    W = 5 * (0.5 - np.random.random(size=(n, d)))
    H = np.random.random(size=(n, d))
    residual = []
    
    for iter in range(max_iters):
        W = V@H@np.linalg.inv(H.T@H)
        
        A = V.T@W
        A_plus, A_ = np.where(A >= 0, A, 0), np.where(A <= 0, -A, 0)
        B = W.T@W
        B_plus, B_ = np.where(B >= 0, B, 0), np.where(B <= 0, -B, 0)
        
        H = H * np.sqrt((A_plus + H@B_) / (A_ + H@B_plus))
        
        residual.append(np.linalg.norm(V-W@H.T, ord='fro'))
    return W, H.T, residual

def kernel_nmf(V, d, kernel='gaussian', sigma=1.0, degree=3, alpha=1.0, beta=1.0):
    """
    Inspired by "Non-negative Matrix Factorization on Kernels", Zhang, Zhou, Chen
    """
    if kernel == 'gaussian':
        A = rbf_kernel(V, gamma=1/sigma**2)
    if kernel == 'polynomial':
        A = polynomial_kernel(V, degree)
    if kernel == 'sigmoid':
        A = sigmoid_kernel(V, gamma=alpha, coef0=beta)
    nm = NMF(n_components=d)
    w = nm.fit_transform(A)
    h = nm.components_
    return w, h, np.linalg.norm(A - w@h)

def rgnmf_multi(X, d, alpha=1.0, beta=1.0, eps=1e-10, max_iters=100):
    """
    Inspired by "Robust Graph Regularized Nonnegative Matrix Factorization for Clustering", Peng, Kang, Cheng, Hu
    https://www.researchgate.net/publication/308718276_Robust_Graph_Regularized_Nonnegative_Matrix_Factorization_for_Clustering
    """
    # Initialization
    n = len(X)
    U = np.random.random(size=(n, d))
    V = np.random.random(size=(n, d))
    S = np.random.random(size=(n, n))
    residual = []
    
    # graph laplacian
    W = np.where(X > 0, -1, 0)
    D = - np.diag(np.sum(W, axis=0))
        
    # update rule
    for iter in range(max_iters):
        U = U * ((X - S)@V) / (U@V.T@V)
        V = V * ((X - S).T@U + beta* W@V) / (V@U.T@U + beta*D@V)
        S = np.where(X - U@V.T >= alpha, X - U@V.T, 0)
        
        residual.append(np.linalg.norm(X - U@V.T - S, ord='fro'))
    
    return U, V.T, S, residual

def sym_nmf(V, d, eps=1e-3, sigma=0.9, beta=0.8, max_iters=100):
    '''
    "Symmetric Nonnegative Matrix Factorization for Graph Clustering", Kuang, Ding, Park
    Implementation according to https://faculty.cc.gatech.edu/~hpark/papers/DaDingParkSDM12.pdf

    '''
    n = len(V)
    h = np.random.random(size=(n, d))
    residual = []
    iter = 0
    while np.linalg.norm(V - h@h.T, ord='fro') > eps and iter < max_iters:
        iter += 1
        S = np.eye(n*d)
        alpha = 1
        grad_f = 4 * (h@h.T - V) @h
        t = h - alpha * grad_f
        h_new = np.where(t >=0, t, 0)
        # while True:
        #     t = h - alpha * grad_f
        #     h_new = np.where(t >=0, t, 0)
        #     if np.linalg.norm(V - h@h.T, ord='fro') - np.linalg.norm(V - h@h.T, ord='fro') - sigma * np.linalg.norm(grad_f.T@(h_new - h), ord='fro') >= -1e-5:
        #         alpha = beta*alpha
        h = h_new
        residual.append(np.linalg.norm(V - h@h.T, ord='fro'))
    return h, h.T, residual
                
