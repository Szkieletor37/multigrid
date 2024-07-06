import numpy as np
import matplotlib.pyplot as plt

import lib, plot

N_FINE = 16
N_COARSE = N_FINE // 2

def convert_latex_matrix(matrix):
    latex_matrix = "\\begin{bmatrix}\n"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] < 1e-10:
                latex_matrix += "0 & "
            else:
                latex_matrix += f"{matrix[i][j]:.3g} & "
        latex_matrix = latex_matrix[:-2] + "\\\\\n"

    latex_matrix += "\\end{bmatrix}"
    print("S(LaTeX): ", latex_matrix)

def experiment_two_grid_iter_matrix(approx_matrix, interpolation_matrix, restriction_matrix, coarse_approx_matrix):

    # A_2h^(-1)
    coarse_approx_matrix_inv = np.linalg.inv(coarse_approx_matrix)

    # S を求める
    two_grid_iter_matrix = interpolation_matrix @ coarse_approx_matrix_inv @ restriction_matrix @ approx_matrix

    # S の固有値を求める
    (two_grid_iter_mat_eigenvalues, two_grid_iter_mat_vectors) = np.linalg.eig(two_grid_iter_matrix)

    np.set_printoptions(precision=4, suppress=True)

    print("-" * 20)
    print("反復行列S: ", two_grid_iter_matrix)
    print("-" * 20)
    convert_latex_matrix(two_grid_iter_matrix)
    print("-" * 20)
    print("S の固有値: ", two_grid_iter_mat_eigenvalues)
    print("-" * 20)
    print("S の固有ベクトル: ", two_grid_iter_mat_vectors)

def main():

    approx_matrix = lib.generate_init_approximation_matrix(N_FINE)
    exact_solution = lib.generate_exact_solution(N_FINE, N_FINE - 1)
    approx_solution = lib.generate_init_approximation_solution(N_FINE - 1)

    init_error = np.linalg.norm(exact_solution - approx_solution)

    plot.plot_exact_solution(exact_solution, "Two-Grid Method, exact", N_FINE)
    plt.figure()
    plot.plot_init_approx_solution(approx_solution, "Two-Grid Method, initial approx", N_FINE)

    # 重み付きヤコビを3回反復して v を出す
    for i in range(3):
        approx_solution = lib.weighted_jacobi_iter(N_FINE, approx_matrix, approx_solution, exact_solution)
        if i == 0:
            plt.figure()
            plot.plot_approximation_solution(approx_solution, "Weighted Jacobi Method, iter = 1", N_FINE)
        elif i == 2:
            plt.figure()
            plot.plot_approximation_solution(approx_solution, "Weighted Jacobi Method, iter = 3", N_FINE)

    # 残差 r = b - Av
    residual = exact_solution - approx_matrix @ approx_solution

    # R, Iを生成
    interpolation_matrix = lib.generate_interpolation_matrix(N_COARSE, N_FINE)
    restriction_matrix = lib.generate_restriction_matrix(interpolation_matrix)

    # r_2h = Rr
    restricted_residual = restriction_matrix @ residual

    # A_2h = RAI
    coarse_approx_matrix = restriction_matrix @ approx_matrix @ interpolation_matrix

    # S についての実験
    experiment_two_grid_iter_matrix(approx_matrix, interpolation_matrix, restriction_matrix, coarse_approx_matrix)

    # E_2h を求める
    init_approx_error = np.zeros(N_COARSE - 1)
    # 重み付きヤコビを3回反復して v を出す
    for _ in range(3):
        approx_error = lib.weighted_jacobi_iter(N_COARSE, coarse_approx_matrix, init_approx_error, restricted_residual)

    # E = IE_2h
    interpolated_approx_error = interpolation_matrix @ approx_error

    # v := v + E
    approx_solution = approx_solution + interpolated_approx_error

    plt.figure()
    plot.plot_approximation_solution(approx_solution, "Two-Grid Method, after interpolated", N_FINE)

    # 再度重み付きヤコビを3回反復する
    for i in range(3):
        approx_solution = lib.weighted_jacobi_iter(N_FINE, approx_matrix, approx_solution, exact_solution)

    plt.figure()
    plot.plot_approximation_solution(approx_solution, "Two-Grid Method, final", N_FINE)

    after_error = np.linalg.norm(exact_solution - approx_solution)

    error_rate = after_error / init_error
    print("Error rate: ", error_rate * 100, "%")

if __name__ == "__main__":
    main()