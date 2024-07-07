import numpy as np

def generate_init_approximation_solution(vec_size):
    init_approx_solution = np.zeros(vec_size)
    for i in range(vec_size):
        # [-1, 1]
        #init_approx_solution[i] = np.random.rand() * 2 - 1
        # [0, 1]
        init_approx_solution[i] = np.random.rand()

    return init_approx_solution

# generate Poisson equation's exact solution
def generate_exact_solution(num_divisions, vec_size, wavenumber):

    print(f"generating_exact_solution: num_divisions: {num_divisions}, vec_size: {vec_size}, wavenumber: {wavenumber}")
    exact_solution = np.zeros(vec_size)
    for i in range(num_divisions + 1):
        if (i == 0) or (i == vec_size+1):
            continue
        else:
            exact_solution[i-1] = np.sin(((i * 1.0) / num_divisions) * wavenumber * np.pi)
    print(f"generated_exact_solution: {exact_solution}")
    
    return exact_solution

def generate_poisson_matrix(num_divisions):
    mat_size = num_divisions - 1
    matrix = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        if i > 0:
            matrix[i][i-1] = -1.0
        matrix[i][i] = 2.0
        if i < mat_size - 1:
            matrix[i][i+1] = -1.0

    h = 1.0 / num_divisions
    poisson_matrix = (1.0 / h**2) * matrix

    return poisson_matrix
# 近似行列を生成
def generate_init_approximation_matrix(num_divisions):
    mat_size = num_divisions - 1
    matrix = np.zeros((mat_size, mat_size))
    #matrix = generate_poisson_matrix(num_divisions)

    for i in range(mat_size):
        if i == 0:
            matrix[i][i] = 2.0
            matrix[i][i+1] = -1.0
        elif i == (mat_size - 1):
            matrix[i][i-1] = -1.0
            matrix[i][i] = 2.0
        else:
            matrix[i][i-1] = -1.0
            matrix[i][i] = 2.0
            matrix[i][i+1] = -1.0
    
    #grid_width = 1.0 / num_divisions

    #approx_matrix = (1.0 / ((grid_width) ** 2)) * matrix

    #return approx_matrix
    return matrix


# 線形補間行列を生成
def generate_interpolation_matrix(num_coarse_divisions, num_fine_divisions):

    # debug
    #print("num_coarse_divisions: ", num_coarse_divisions)
    #print("num_fine_divisions: ", num_fine_divisions)

    # 両端の点は除く
    row_size = num_fine_divisions - 1
    col_size = num_coarse_divisions - 1

    matrix = np.zeros((row_size, col_size))

    for i in range(col_size):
        matrix[2*i][i] = 0.5
        matrix[2*i+1][i] = 1.0
        matrix[2*i+2][i] = 0.5

    return matrix

def generate_restriction_matrix(interpolation_matrix):
    return 0.5 * interpolation_matrix.T

def weighted_jacobi_iter(num_divisions, approx_matrix, init_approx_solution, scaled_rhs):


    print('-' * 20)
    print("Jacobi iter starting...")
    print("N: ", num_divisions)
    print("A: ", approx_matrix)
    print("v_0: ", init_approx_solution)
    print("b: ", scaled_rhs)

    mat_size = num_divisions - 1

    # D^-1
    diag_approx_matrix = np.diag(np.diag(approx_matrix))
    inv_diag_approx_matrix = np.linalg.inv(diag_approx_matrix)
    print("D^-1: ", inv_diag_approx_matrix)

    # L
    strictly_lower_approx_matrix = -1.0 * np.tril(approx_matrix, k=-1)
    # U
    strictly_upper_approx_matrix = -1.0 * np.triu(approx_matrix, k=1)
    #print("D - L - U: ", diag_approx_matrix - strictly_lower_approx_matrix - strictly_upper_approx_matrix)
    print("L + U: ", strictly_lower_approx_matrix + strictly_upper_approx_matrix)

    #print("D^-1(L + U): ", inv_diag_approx_matrix @ (strictly_lower_approx_matrix + strictly_upper_approx_matrix))

    #print("h^2 * D^-1b: ", num_divisions * num_divisions * inv_diag_approx_matrix @ exact_solution)
    print("D^-1b: ", inv_diag_approx_matrix @ scaled_rhs)

    # ω
    weight = 2.0 / 3.0
    print("weight: ", weight)

    # (1 - ω) * I + ω * D^-1 * (L + U)
    weighted_jacobi_matrix = (1.0 - weight) * np.eye(mat_size) + weight * inv_diag_approx_matrix @ (strictly_lower_approx_matrix + strictly_upper_approx_matrix)
    #jacobi_matrix = inv_diag_approx_matrix @ (strictly_lower_approx_matrix + strictly_upper_approx_matrix)



    print("Mv: ", weighted_jacobi_matrix @ init_approx_solution)

    #print("Jacobi matrix: ", jacobi_matrix)

    #for i in range(mat_size):
    #    if i == 0:
    #        jacobi_matrix[i][i+1] = 1.0
    #    elif i == (mat_size - 1):
    #        jacobi_matrix[i][i-1] = 1.0
    #    else:
    #        jacobi_matrix[i][i-1] = 1.0
    #        jacobi_matrix[i][i+1] = 1.0


    # (I - ω * D^-1 * A) * v + ω * D^-1 * b
    #approx_solution = (np.eye(mat_size) - (weight * 0.5 * jacobi_matrix)) @ init_approx_solution + weight * 0.5 * exact_solution
    #approx_solution = jacobi_matrix @ init_approx_solution + num_divisions * num_divisions * weight * inv_diag_approx_matrix @ exact_solution
    #weighted_approx_solution = weighted_jacobi_matrix @ init_approx_solution + weight * inv_diag_approx_matrix @ scaled_rhs
    #approx_solution = jacobi_matrix @ init_approx_solution + inv_diag_approx_matrix @ exact_solution
    #weighted_approx_solution = init_approx_solution + weight * (inv_diag_approx_matrix @ (scaled_rhs - (strictly_lower_approx_matrix + strictly_upper_approx_matrix) @ init_approx_solution))
    weighted_approx_solution = init_approx_solution + weight * (inv_diag_approx_matrix @ (scaled_rhs - approx_matrix @ init_approx_solution))


    print("Jacobi iter ending...")
        
    return weighted_approx_solution