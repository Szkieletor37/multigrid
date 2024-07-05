
import numpy as np

import plot

# N = 2^10
# POW >= 2 (1だと動かない)
POW = 2
N = 2 ** POW



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

# Au = b の近似解 v を求める
# multigrid(N, A, b) -> v
def multigrid(num_divisions, approx_matrix, init_approx_solution, exact_solution):
    print('-' * 20)
    print("multigrid cycle starting...")
    print("N: ", num_divisions)
    print("A: ", approx_matrix)
    print("v_0: ", init_approx_solution)
    print("b: ", exact_solution)
    print('-' * 20)

    # N == 2 なら終了
    if num_divisions == 2:
        return init_approx_solution
    
    approx_solution = init_approx_solution

    # 重み付きヤコビを3回反復して v を出す
    for i in range(3):
        approx_solution = weighted_jacobi_iter(num_divisions, approx_solution)

    # 残差 r = b - Av
    residual = exact_solution - approx_matrix @ approx_solution

    # R, Iを生成
    interpolation_matrix = generate_interpolation_matrix(num_divisions // 2, num_divisions)
    restriction_matrix = generate_restriction_matrix(interpolation_matrix)

    # r_2h = Rr
    restricted_residual = restriction_matrix @ residual

    # A_2h = RAI
    fine_approx_matrix = restriction_matrix @ approx_matrix @ interpolation_matrix

    # E_2h を求める
    approx_error = multigrid(num_divisions // 2, fine_approx_matrix, np.zeros(num_divisions // 2 - 1), restricted_residual)

    # E = IE_2h
    interpolated_approx_error = interpolation_matrix @ approx_error

    # v := v + E
    approx_solution = approx_solution + interpolated_approx_error

    # 再度重み付きヤコビを3回反復する
    for i in range(3):
        approx_solution = weighted_jacobi_iter(num_divisions, approx_solution)

    return approx_solution

def main():
    # 近似解の初期値はランダムとする
    mat_size = N - 1
    init_approx_solution = np.zeros(mat_size)
    for i in range(mat_size):
        init_approx_solution[i] = np.random.rand()


    init_approx_matrix = generate_init_approximation_matrix(N)

    exact_solution = np.zeros(mat_size)
    for i in range(N+1):
        if (i == 0) or (i == mat_size+1):
            continue
        else:
            exact_solution[i-1] = np.sin(((i * 1.0) / N) * np.pi)

    print("init_approx_solution: ", init_approx_solution)
    print("init_approx_matrix: ", init_approx_matrix)
    print("exact_solution: ", exact_solution)
    print("---")
    print("Plotting exact solution...")
    plot.plot_exact_solution(exact_solution)

    ans = multigrid(N, init_approx_matrix, init_approx_solution, exact_solution)

    print("Answer: ", ans)

if __name__ == "__main__":
    main()