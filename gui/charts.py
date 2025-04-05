import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def draw_charts(history, avg_history, std_history, config=None, best_fitness=None, solution=None):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 7))
    spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1.5, 4], height_ratios=[1, 1])

    # Panel z parametrami
    if config:

        if best_fitness is not None:
            fitness_line = f"Best fitness: {best_fitness:.5f}"
        else:
            fitness_line = "Best fitness: --"

        if solution is not None:
            solution_line = f"Solution: {np.array2string(np.array(solution), precision=3, separator=', ')}"
        else:
            solution_line = "Solution: --"

        param_text = (
            f"Parametry uruchomienia:\n"
            f"Function: {config.get('function', '-')}\n"
            f"Minimize: {config['minimize']}\n"
            f"Precision: {config['precision']}\n"
            f"Num epochs: {config['num_epochs']}\n"
            f"Population size: {config['population_size']}\n"
            f"Selection method: {config['selection_method']}\n"
            f"Crossover method: {config['crossover_method']}\n"
            f"Crossover prob: {config['crossover_prob']}\n"
            f"Mutation method: {config['mutation_method']}\n"
            f"Mutation prob: {config['mutation_prob']}\n"
            f"Inversion prob: {config['inversion_prob']}\n"
            f"Elitism rate: {config['elitism_rate']}\n"
            f"Num variables: {config['num_variables']}\n"
            f"{fitness_line}\n"
            f"{solution_line}"
        )

        ax_text = fig.add_subplot(spec[:, 0])
        ax_text.axis('off')
        ax_text.text(0, 1, param_text, fontsize=10, va='top', fontfamily='monospace')

    # Wykresy
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.plot(history, color='blue')
    ax1.set_title("Najlepsza wartość w każdej epoce")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Fitness")

    ax2 = fig.add_subplot(spec[1, 1])
    ax2.plot(avg_history, label='Średnia', color='dodgerblue')
    ax2.plot(std_history, label='Odchylenie std.', color='orange')
    ax2.set_title("Statystyki populacji")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Wartość")
    ax2.legend()

    plt.tight_layout()
    plt.show()

