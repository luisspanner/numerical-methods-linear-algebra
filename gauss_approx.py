"""
TODO: Implementierung der Gauß-Approximation
- Skalarprodukt und Norm
- Legendre-Polynome
- Probe
- Koeffizienten der Approximationspolynome
- Auswertung der Approximationspolynome
- Plotten
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys

def Skalarprodukt(f, g, a, b):
    return integrate.quad(lambda x: f(x) * g(x), a, b)[0]

def Norm(f, a, b):
    return np.sqrt(Skalarprodukt(f, f, a, b))

def Variablentransformation(f, a, b):
    # Variablentransformation für f von [-1;1] nach x in [a;b]
    return lambda x, f=f, a=a, b=b, c=2/(b-a), wc = np.sqrt(2/(b-a)): wc * f(-1 + c * (x-a))

def LegendrePolynome(n, a, b):
    L = [None] * n
    # Orthogonale Polynomem auf [-1, 1] mit Rekursionsformel
    L[0] = lambda x: x**0
    L[1] = lambda x: x
    for k in range(2, n):
        coeff = (k-1)**2 / (4 * (k-1)**2 - 1)
        L[k] = lambda x, k = k, coeff = coeff: x * L[k-1](x) - coeff * L[k-2](x)
    
    # Normierte Polynome
    phi = [None] * n
    for k in range(n):
        norm = Norm(L[k], -1, 1)
        phi[k] = lambda x, k=k, norm=norm: L[k](x) / norm

    # Variablentransformation
    if a != -1 or b != 1:
        for k in range(n):
            phi[k] = Variablentransformation(phi[k], a, b)

    return phi

def ProbeLegendrePolynome(phi, a, b):
    n = len(phi)
    for k in range(n):
        print(f"||phi{k}|| = {Norm(phi[k], a, b)}")
    for k in range(n):
        for j in range(k+1, n):
            print(f"<phi{k}, phi{j}> = {Skalarprodukt(phi[k], phi[j], a, b)}")

def Koeffizienten(f, phi, a, b):
    n = len(phi)
    alpha = np.empty(n, dtype=np.double)
    for k in range(n):
        alpha[k] = Skalarprodukt(f, phi[k], a, b)
    print("alpha =", alpha)
    return alpha

def Approximationspolynome(alpha, phi):
    n = len(alpha)
    p = [None] * n
    p[0] = lambda x: alpha[0] * phi[0](x)
    for k in range(1, n):
        p[k] = lambda x, k=k: p[k-1](x) + alpha[k] * phi[k](x)
    return p
    
def Auswertung(f, p, a, b):
    #Auswertung von f und den Polynomen auf einem Linspace
    n = len(p)
    x = np.linspace(a-0.5, b+0.5, 100)
    yf = f(x)
    y = [None] * n
    for i in range(n):
        y[i] = p[i](x)
    return x, yf, y

def plot(x, yf, y):
    n = len(y)
    style = ["k-", "r-", "g-", "c-", "m"]
    args = [None] * (3 *(1 + n))
    legend = [None] * (1 + n)
    args[0:3] = [x, yf, "b-"]
    legend[0] = "f(x)"
    for k in range(n):
        args[3*(k + 1) : 3 * (k + 2)] = [x, y[k], style[k%5]]
        legend[k + 1] = f"p{k}(x)"
    plt.plot(*args)
    plt.legend(legend)
    plt.grid()
    plt.savefig("Approximation.png")
    plt.show()

def main():
    Probe = True
    n = int(sys.argv[1] if len(sys.argv) > 1 else 5)
    a = int(sys.argv[2] if len(sys.argv) > 1 else -1)
    b = int(sys.argv[3] if len(sys.argv) > 1 else 1)

    f = lambda x: 1.5 * np.sin(np.pi * (x-0.2)) + 0.3
    phi = LegendrePolynome(n, a, b)
    if Probe:
        ProbeLegendrePolynome(phi, a, b)

    alpha = Koeffizienten(f, phi, a, b)
    p = Approximationspolynome(alpha, phi)
    x, yf, y = Auswertung(f, p, a, b)
    plot(x, yf, y)


if __name__ == "__main__":
    main()