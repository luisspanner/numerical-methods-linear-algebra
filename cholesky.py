import numpy as np

def cholesky(A):
    m, n  = A.shape
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric and hence Cholesky decomposition cannot be built")
    L = np.zeros((n, n))
    # Erster Diagonaleintrag
    L[0, 0] = np.sqrt(A[0, 0])

    # Fülle erste Spalte
    for j in range(1, n):
        L[j, 0] = A[j ,0] / L[0, 0]
    
    for i in range(1, n):
        L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i] ** 2))
    
        for j in range(i + 1, n):
            s = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - s) / L[i, i]
    
    return L


def main():
    A = np.array([[5.0, -5.0, 0.0, 0.0], \
                 [-5.0, 7.0, -2.0, 0.0], \
                 [0.0, -2.0, 20.0, -18.0], \
                 [0.0, 0.0, -18.0, 19.0]])
    #b = np.array([5.0, -7.0, 20.0, -17.0])

    cholesky_factor = cholesky(A)
    print("Cholesky-Faktor:")
    print(cholesky_factor)


if __name__ == "__main__":
    main()