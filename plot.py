import numpy as np
import matplotlib.pyplot as plt

def plot_exact_solution(exact_solution, graph_title, num_divisions):
    
    # 最初に0を追加する
    exact_solution = np.insert(exact_solution, 0, 0)

    # 最後に0を追加する
    exact_solution = np.append(exact_solution, 0)

    x = np.arange(0, 1, 1.0 / len(exact_solution))
    y = exact_solution
    plt.title(graph_title + ', N = ' + str(num_divisions))
    plt.xlabel('x')
    plt.ylabel('value')
    plt.ylim(0, 1)
    plt.plot(x, y)
    plt.legend(['exact solution'])
    plt.savefig('img/ex_sol_' + graph_title + '.png')
    print('---')
    print('Exact solution plot saved as img/ex_sol_' + graph_title + '.png')

def plot_init_approx_solution(init_approx_solution, graph_title, num_divisions):
    # 最初に0を追加する
    init_approx_solution = np.insert(init_approx_solution, 0, 0)

    # 最後に0を追加する
    init_approx_solution = np.append(init_approx_solution, 0)

    x = np.arange(0, 1, 1.0 / len(init_approx_solution))
    y = init_approx_solution
    plt.title(graph_title + ', N = ' + str(num_divisions))
    plt.xlabel('x')
    plt.ylabel('value')
    plt.ylim(0, 1)
    plt.plot(x, y)
    plt.legend(['initial approximation solution'])
    plt.savefig('img/init_approx_sol_' + graph_title + '.png')
    print('---')
    print('Initial solution plot saved as img/init_approx_sol_' + graph_title + '.png')

def plot_approximation_solution(approx_solution, graph_title, num_divisions):
    # 最初に0を追加する
    approx_solution = np.insert(approx_solution, 0, 0)

    # 最後に0を追加する
    approx_solution = np.append(approx_solution, 0)
    x = np.arange(0, 1, 1.0 / len(approx_solution))
    y = approx_solution
    plt.title(graph_title + ', N = ' + str(num_divisions))
    plt.xlabel('x')
    plt.ylabel('value')
    plt.ylim(0, 1)
    plt.plot(x, y)
    plt.legend(['approximation solution'])

    graph_title = graph_title.replace(' ', '_')
    plt.savefig('img/approx_sol_' + graph_title + '.png')
    print('---')
    print('Approximation solution plot saved as img/approx_sol_' + graph_title + '.png')