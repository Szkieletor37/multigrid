import numpy as np

import lib

N_FINE = 32
N_COARSE = N_FINE // 2

def main():

    approx_matrix = lib.generate_init_approximation_matrix(N_FINE)
    exact_solution = lib.generate_exact_solution(N_FINE, N_FINE - 1)
    approx_solution = lib.generate_init_approximation_solution(N_FINE - 1)

    # 重み付きヤコビを3回反復して v を出す
    for _ in range(3):
        approx_solution = lib.weighted_jacobi_iter(N_FINE, approx_solution)

    # 残差 r = b - Av
    residual = exact_solution - approx_matrix @ approx_solution

    # R, Iを生成
    interpolation_matrix = lib.generate_interpolation_matrix(N_COARSE, N_FINE)
    restriction_matrix = lib.generate_restriction_matrix(interpolation_matrix)

    # r_2h = Rr
    restricted_residual = restriction_matrix @ residual

    # A_2h = RAI
    fine_approx_matrix = restriction_matrix @ approx_matrix @ interpolation_matrix

    # E_2h を求める
    approx_error = np.zeros(N_COARSE - 1)
    # 重み付きヤコビを3回反復して v を出す
    for _ in range(3):
        approx_error = lib.weighted_jacobi_iter(N_COARSE, approx_error)

    # E = IE_2h
    interpolated_approx_error = interpolation_matrix @ approx_error

    # v := v + E
    approx_solution = approx_solution + interpolated_approx_error

    # 再度重み付きヤコビを3回反復する
    for i in range(3):
        approx_solution = lib.weighted_jacobi_iter(N_FINE, approx_solution)

    return approx_solution

if __name__ == "__main__":
    main()