import numpy as np
import matplotlib.pyplot as plt

def plot_exact_solution(exact_solution):
    x = np.arange(0, 1, 1.0 / len(exact_solution))
    y = exact_solution
    plt.xlabel('x')
    plt.ylabel('value')
    plt.plot(x, y)
    plt.savefig('img/ex_sol.png')
    print('---')
    print('Exact solution plot saved as img/ex_sol.png')
