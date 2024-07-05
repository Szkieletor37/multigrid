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

def weighted_jacobi_iter(num_divisions, init_approx_solution):

    print('-' * 20)
    print("Jacobi iter starting...")
    print("N: ", num_divisions)
    print("v_0: ", init_approx_solution)
    print('-' * 20)

    mat_size = num_divisions - 1

    approx_solution = np.zeros(mat_size)
    weight = 2.0 / 3.0

    jacobi_matrix = np.zeros((mat_size, mat_size))

    for i in range(mat_size):
        if i == 0:
            jacobi_matrix[i][i+1] = 0.5
        elif i == (mat_size - 1):
            jacobi_matrix[i][i-1] = 0.5
        else:
            jacobi_matrix[i][i-1] = 0.5
            jacobi_matrix[i][i+1] = 0.5

    intermediate_solution = jacobi_matrix @ init_approx_solution
    approx_solution = (1.0 - weight) * init_approx_solution + weight * intermediate_solution
        
    return approx_solution

