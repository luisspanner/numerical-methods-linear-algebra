def main():
    n = get_matrix_size()
    A = get_input_matrix(n)
    b = get_input_vector(n)

    # Building A|b from A and b
    Ab = [row[:] + [b[i]] for i, row in enumerate(A)]
    R = gaussian_elim_basic(Ab, n)
    x = backwards_substitution(R, n)
    print(f"Solution vector x: {x}")
    print("Transformed Matrix A|b in row echelon form:")
    for i in range(n):
        print(R[i])

def gaussian_elim_basic(Ab, n):
    # Forward elimination
    for i in range(n):
        #Find Pivot
        if Ab[i][i] == 0: # if pivot is 0, swap rows
            for j in range(i + 1, n):
                if Ab[j][i] != 0:
                    temp = Ab[i]
                    Ab[i] = Ab[j]
                    Ab[j] = temp
                    break
        pivot = Ab[i][i]
        if pivot == 0:
            raise ValueError("Linear system of equations does not have a unique solution because A|b does not have full rank!")
        # for every row below
        for j in range(i + 1, n):
            # calculate elimination factor m
            m = Ab[j][i] / pivot
            for k in range(n + 1):
                Ab[j][k] -= m * Ab[i][k]
    return Ab

def gaussian_elim_for_LR(A, n):
    L = unit_matrix(n) # matrix L of LR-decomposition will be built while bringing A in row echelon form, default is unit matrix
    p = [i for i in range(n)] # permutation vector tracks column pivoting
    # Forward elimination
    for i in range(n):
        # Find Pivot
        if A[i][i] == 0: # if pivot is 0, swap rows
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    temp = A[i]
                    temp_p = i
                    A[i] = A[j]
                    p[i] = j
                    A[j] = temp
                    p[j] = temp_p
                    break
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Linear system of equations does not have a unique solution because A|b does not have full rank!")
        # for every row below
        for j in range(i + 1, n):
            # calculate elimination factor m
            m = A[j][i] / pivot
            L[j][i] = m # build matrix L with elimination factors m
            for k in range(n):
                A[j][k] -= m * A[i][k]
    return A, L, p
    
def backwards_substitution(Ab, n):
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i][-1] - sum(Ab[i][j] * x[j] for j in range(i+1, n))) / Ab[i][i]
    return x

def get_matrix_size():
    while True:
        try:
            n = int(input("How many rows should your matrix have? "))
        except ValueError as e:
            print(f"Invalid input: {e}. Try again.")
        else:
            return n

def get_input_matrix(n):
     while True:
        try:
            A = []
            for i in range(n):
                row = list(map(float, input(f"Enter row {i + 1} (space-separated, for example '1 2 3'): ").split()))
                if len(row) != n:
                    raise ValueError(f"Row {i + 1} must have exactly {n} elements.")
                A.append(row)
            return A
        except ValueError as e:
            print(f"Invalid input: {e}")

def get_input_vector(n):
    while True:
            try:
                b = list(map(float, input("Enter vector b: ").split()))
                if len(b) != n:
                    print("Length of vector b must equal number of rows of matrix A. Please enter b again.")
                    continue
                return b
            except ValueError as e:
                print(f"Invalid input: {e}. Try again.")

def unit_matrix(n):
    I = []
    for i in range(n): # default L is unit matrix
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        I.append(row)
    return I

if __name__ == "__main__":
    main()