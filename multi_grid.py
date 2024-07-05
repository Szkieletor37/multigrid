
import numpy as np

import plot
import lib

# N = 2^10
# POW >= 2 (1だと動かない)
POW = 2
N = 2 ** POW

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
        approx_solution = lib.weighted_jacobi_iter(num_divisions, approx_solution)

    # 残差 r = b - Av
    residual = exact_solution - approx_matrix @ approx_solution

    # R, Iを生成
    interpolation_matrix = lib.generate_interpolation_matrix(num_divisions // 2, num_divisions)
    restriction_matrix = lib.generate_restriction_matrix(interpolation_matrix)

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
        approx_solution = lib.weighted_jacobi_iter(num_divisions, approx_solution)

    return approx_solution

def main():
    # 近似解の初期値はランダムとする
    mat_size = N - 1
    init_approx_solution = lib.generate_init_approximation_solution(mat_size)

    init_approx_matrix = lib.generate_init_approximation_matrix(N)

    exact_solution = lib.generate_exact_solution(N, mat_size)

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