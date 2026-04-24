from gauss import get_input_matrix, get_input_vector, get_matrix_size, gaussian_elim_for_LR

def main():
    n = get_matrix_size()
    A = get_input_matrix(n)
    b = get_input_vector(n)
    R, L, p = gaussian_elim_for_LR(A, n)
    c = [b[i] for i in p]
    y = forward_substitution(L, c, n)
    x = backwards_substitution_LR(R, y, n)
    print("LR-Decomposition:")
    print("L:")
    for i in range(n):
        print(L[i])
    print()
    print("R:")
    for i in range(n):
        print(R[i])
    print(f"Solution vector x of linear system of equations Rx = y: {x}")
    print("Note: R is A transformed in row echelon form, y = L^-1 * c " \
        "where c = p * b and p is the permutation vector")
    
def forward_substitution(L, c, n):
    y = [0 for _ in range(n)]
    for i in range(n):
        summe = sum(L[i][j] * y[j] for j in range(i))
        y[i] =  c[i] - summe
    return y

def backwards_substitution_LR(R, y, n):
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(R[i][j] * x[j] for j in range(i+1, n))) / R[i][i]
    return x

if __name__ == "__main__":
    main()