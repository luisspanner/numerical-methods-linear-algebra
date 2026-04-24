import numpy as np

def LRkompakt(A, options):
    #Initialisierung
    m, n = A.shape
    r = -1
    Stufen = []
    p = np.array(range(m))
    v = 0
    # Schleife über alle Spalten
    for j in range(n):
        # Pivotsuche
        if r >= m-1:
            continue
        col = abs(A[r+1:m,j])
        i = col.argmax()
        if col[i] <= options["eps"]:
            continue
        r += 1
        i += r
        Stufen.append(j)

        # Zeilenvertauschung
        if i != r:
            v += 1
            A[[i, r]] = A[[r, i]]
            p[[i, r]] = p[[r, i]]

        #Elimination
        for i in range(r+1, m):
            lam = - A[i,j] / A[r,j]
            A[i,j] = 0.0
            A[i,j+1:] += lam * A[r, j+1:]
            A[i, r] = -lam

    return A, p, r + 1, Stufen, v

def LRZerlegung(A, options):
    # Aufruf der LR-Zerlegung mit kompakter Speicherung
    m, n = A.shape
    A, p, Rang, Stufen, v = LRkompakt(A, options)

    # Extrahiere R
    R = np.zeros((m,n))
    for i in range(Rang):
        R[i, Stufen[i]:] = A[i, Stufen[i]:]

    # Extrahiere L
    L = np.eye(m)
    for j in range(Rang):
        L[j+1:,j] = A[j+1:,j]

    # Bilde P
    P = np.eye(m)[p]

    return R, L, P, Rang, Stufen, v

def Vorwaertsloesen(L, c, diag1 = False):
    m = L.shape[0]
    y = np.zeros(m)
    for i in range(m):
        y[i] = c[i] - L[i,:i] @ y[:i]
        if not diag1:
            y[i] /= L[i][i]
    return y

def Rueckwaertsloesen(R, Stufen, d, options):
    m, n = R.shape
    r = len(Stufen)
    # Überprüfe die Lösbarkeit
    if r < m and np.linalg.norm(d[r+1:], np.inf) > options["eps"]:
        x = np.full(n, np.nan)
        K = np.full((n, n-r), np.nan)
        return x, K
    # Berechne die Lösung
    x = np.zeros(n)
    K = np.zeros((n,n-r))
    k = -1
    i = -1
    for j in reversed(range(n)):
        if j == Stufen[i]:
            x[j] = (d[i] - R[i,j+1:] @ x[j+1:]) / R[i,j]
            K[j,:] = (- R[i,j+1:] @ K[j+1:,:]) / R[i,j]
            i -= 1
        else :
            k += 1
            x[j] = 0.0 # nicht notwendig
            K[j, k] = 1.0

    return x, K

def main():
    A = np.array( [ [ 0.0, 2.0, 4.0, 2.0 ], \
                    [ 3.0, 3.0, 3.0, 3.0 ], \
                    [ 1.0, 2.0, 3.0, 1.0 ] ] )
    b = np.array( [ 4, 3, 3 ] )

    options = {"eps" : 1.0e-12}

    R, L, P, Rang, Stufen, v = LRZerlegung(A.copy(), options)
    print("A = "); print(A)
    print("R = "); print(R)
    print("L = "); print(L)
    print("P = "); print(P)
    print("Stufen = ", Stufen)
    print("Rang = ", Rang)
    print("v = ", v)
    print("Probe: ", (np.linalg.norm(L @ R - P @ A, np.inf)))

    y = Vorwaertsloesen(L, P @ b, diag1=True)
    print("y = ", y)
    x, K = Rueckwaertsloesen(R, Stufen, y, options)
    if np.isnan(x[0]):
        print("unlösbar")
    else:
        print("x =", x)
        if K.shape[1] > 0:
            print("Kern: Span von "); print(K)
        print("Probe: ", np.linalg.norm(A @ x - b, np.inf),
                         np.linalg.norm(A @ K, np.inf))

if __name__ == "__main__":
    main()