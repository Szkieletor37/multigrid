
import numpy as np
import matplotlib.pyplot as plt

import plot
import lib

# N = 2^10
# POW >= 2 (1だと動かない)
POW = 10
N = 2 ** POW
JACOBI_ITER_NUM = 3

# Au = b の近似解 v を求める
# multigrid(N, A, b) -> v
def multigrid(num_divisions, approx_matrix, init_approx_solution, scaled_rhs_vector):
    # N <= 2 なら厳密求解
    if num_divisions <= 2:
        approx_solution = np.linalg.solve(approx_matrix, scaled_rhs_vector)
        print("-" * 20)
        print("N <= 2, Solve exactly...")
        print("N: ", num_divisions)
        print("A: ", approx_matrix)
        print("b: ", scaled_rhs_vector)
        print("v: ", approx_solution)
        print('-' * 20)
        return approx_solution

    #print('-' * 20)
    #print("multigrid cycle starting...")
    #print("N: ", num_divisions)
    #print("A: ", approx_matrix)
    #print("v_0: ", init_approx_solution)
    #print("b: ", exact_solution)
    #print('-' * 20)
    
    approx_solution = init_approx_solution

    # 重み付きヤコビを定数回反復して v を出す
    print('-' * 20)
    print(f"Pre-smoothing: Iteration for {JACOBI_ITER_NUM} times ...")
    print("approx_solution (before): ", approx_solution)
    print("scaled_rhs_vector: ", scaled_rhs_vector)
    for i in range(JACOBI_ITER_NUM):
        print(f"Iteration {i + 1} ...")
        approx_solution = lib.weighted_jacobi_iter(num_divisions, approx_matrix, approx_solution, scaled_rhs_vector)
        print("approx_solution: ", approx_solution)

    if num_divisions == N:
        plt.figure()
        plot.plot_approximation_solution(approx_solution, "Multi-Grid Method, pre-smoothing", N)

    # 残差 r = b - Av
    residual = scaled_rhs_vector - approx_matrix @ approx_solution

    # R, Iを生成
    interpolation_matrix = lib.generate_interpolation_matrix(num_divisions // 2, num_divisions)
    restriction_matrix = lib.generate_restriction_matrix(interpolation_matrix)

    # r_2h = Rr
    restricted_residual = restriction_matrix @ residual

    # A_2h = RAI
    coarse_approx_matrix = restriction_matrix @ approx_matrix @ interpolation_matrix

    coarse_num_divisions = num_divisions // 2

    print('-' * 20)
    print(f"Move to Coarse (N = {coarse_num_divisions}) Grid...")
    print("A_2h: ", coarse_approx_matrix)
    print("r_2h: ", restricted_residual)
    
    #init_coarse_approx_solution = lib.generate_init_approximation_solution(coarse_num_divisions - 1)
    init_coarse_approx_solution = np.zeros(coarse_num_divisions - 1)

    # E_2h を求める
    # TODO: これ合ってる？ (反復はしない？)
    approx_error = multigrid(coarse_num_divisions, coarse_approx_matrix, init_coarse_approx_solution, restricted_residual)

    print(f"\nReturn to Fine (N = {num_divisions}) Grid...")

    # E = IE_2h
    interpolated_approx_error = interpolation_matrix @ approx_error
    print("interpolated_approx_error: ", interpolated_approx_error)

    # v := v + E
    approx_solution = approx_solution + interpolated_approx_error
    print("approx_solution (after): ", approx_solution)

    if num_divisions == N:
        plt.figure()
        plot.plot_approximation_solution(approx_solution, "Multi-Grid Method, interpolated", N)

    # 再度重み付きヤコビを定数回反復する
    print('-' * 20)
    print(f"Post-smoothing: Iteration for {JACOBI_ITER_NUM} times ...")
    print("approx_solution (before): ", approx_solution)
    print("scaled_rhs_vector: ", scaled_rhs_vector)
    for i in range(JACOBI_ITER_NUM):
        print(f"Iteration {i + 1} ...")
        approx_solution = lib.weighted_jacobi_iter(num_divisions, approx_matrix, approx_solution, scaled_rhs_vector)
        print("approx_solution: ", approx_solution)

    return approx_solution

def calc_error_rate(init_approx_solution, final_approx_solution, exact_solution):
    print("Calculating error rate...")
    print("init_approx_solution: ", init_approx_solution)
    print("final_approx_solution: ", final_approx_solution)
    print("exact_solution: ", exact_solution)
    init_error = np.linalg.norm(exact_solution - init_approx_solution)
    after_error = np.linalg.norm(exact_solution - final_approx_solution)

    error_rate = after_error / init_error
    print("Error rate: ", error_rate * 100, "%")

def main():
    # 近似解の初期値はランダムとする
    mat_size = N - 1
    init_approx_solution = lib.generate_init_approximation_solution(mat_size)

    init_approx_matrix = lib.generate_init_approximation_matrix(N)

    exact_solution = lib.generate_exact_solution(N, mat_size, 1)
    scaled_rhs_vector = init_approx_matrix @ exact_solution

    print("init_approx_solution: ", init_approx_solution)
    print("init_approx_matrix: ", init_approx_matrix)
    print("exact_solution: ", exact_solution)
    print("---")
    print("Plotting exact solution...")
    plot.plot_exact_solution(exact_solution, "Multi-Grid Method, exact", N)

    print("Plotting initial approximation solution...")
    plt.figure()
    plot.plot_approximation_solution(init_approx_solution, "Multi-Grid Method, initial approx", N)

    final_approx_solution = multigrid(N, init_approx_matrix, init_approx_solution, scaled_rhs_vector)

    print("Plotting final approximation solution...")
    plt.figure()
    plot.plot_approximation_solution(final_approx_solution, "Multi-Grid Method, final", N)

    calc_error_rate(init_approx_solution, final_approx_solution, exact_solution)

    return 0


if __name__ == "__main__":
    main()