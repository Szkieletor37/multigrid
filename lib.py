import numpy as np

def generate_init_approximation_solution(vec_size):
    init_approx_solution = np.zeros(vec_size)
    for i in range(vec_size):
        init_approx_solution[i] = np.random.rand()

    return init_approx_solution

def generate_exact_solution(num_divisions, vec_size):
    exact_solution = np.zeros(vec_size)
    for i in range(num_divisions + 1):
        if (i == 0) or (i == vec_size+1):
            continue
        else:
            exact_solution[i-1] = np.sin(((i * 1.0) / num_divisions) * np.pi)
    
    return exact_solution

# 近似行列を生成
def generate_init_approximation_matrix(num_divisions):
    mat_size = num_divisions - 1
    matrix = np.zeros((mat_size, mat_size))

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
    
    grid_width = 1.0 / num_divisions

    approx_matrix = (1.0 / ((grid_width) ** 2)) * matrix

    return approx_matrix


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

def weighted_jacobi_iter(num_divisions, approx_matrix, init_approx_solution, exact_solution):

    #print('-' * 20)
    #print("Jacobi iter starting...")
    #print("N: ", num_divisions)
    #print("A: ", approx_matrix)
    #print("v_0: ", init_approx_solution)
    #print('-' * 20)

    mat_size = num_divisions - 1

    # D^-1
    diag_approx_matrix = np.diag(np.diag(approx_matrix))
    inv_diag_approx_matrix = np.linalg.inv(diag_approx_matrix)

    # v の初期値
    approx_solution = np.zeros(mat_size)

    # ω
    weight = 2.0 / 3.0

    #jacobi_matrix = np.zeros((mat_size, mat_size))
    jacobi_matrix = np.eye(mat_size) - weight * inv_diag_approx_matrix @ approx_matrix

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
    approx_solution = jacobi_matrix @ init_approx_solution + weight * inv_diag_approx_matrix @ exact_solution
        
    return approx_solution