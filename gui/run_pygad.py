import pygad, numpy as np, time, csv, matplotlib.pyplot as plt
from algorithm.ga_operators import (one_point_crossover, two_point_crossover,
                          uniform_crossover, mutation_swap,
                          mutation_random_bit, arithmetic_crossover,
                          mutation_gauss)
import algorithm.fitness_wrappers as fit

# ------- konfiguracje testowe --------------------------------------
EXPERIMENTS = [
    # (label, fitness, gene_type, n_genes, init_low, init_high,
    #  selection, crossover_fn, mutation_fn)
    ("bin_tourn_1pt_swap", fit.fitness_hyper_bin, int, 200, 0, 2,
     "tournament", one_point_crossover, mutation_swap),

    ("bin_rws_uniform_randbit", fit.fitness_hyper_bin, int, 200, 0, 2,
     "rws", uniform_crossover, mutation_random_bit),

    ("real_rand_arith_gauss", fit.fitness_hyper_real, float, 10,
     -32.768, 32.768, "random", arithmetic_crossover, mutation_gauss),
]

# ---------------- helper: run single GA ----------------------------
def run_once(label, fitness_func, gene_type, n_genes, low, high,
             sel, xover, mut, runs=30):
    best_all = np.empty((runs,))
    mean_all = np.empty((runs,))

    for i in range(runs):
        ga = pygad.GA(num_generations=300,
                      sol_per_pop=50,
                      num_parents_mating=20,
                      num_genes=n_genes,
                      fitness_func=fitness_func,
                      gene_type=gene_type,
                      init_range_low=low,
                      init_range_high=high,
                      parent_selection_type=sel,
                      crossover_type=xover,
                      mutation_type=mut,
                      keep_elitism=1,
                      K_tournament=3)

        t0 = time.time()
        ga.run()
        t1 = time.time()

        best_all[i] = ga.best_solution()[1]
        mean_all[i] = np.mean(ga.best_solutions_fitness)
        print(f"{label} run {i+1}/{runs}: best={best_all[i]:.4f}  time={t1-t0:.2f}s")

    # --- zapisz CSV -------------------------------------------------
    with open(f"{label}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "best", "mean"])
        for idx,(b,m) in enumerate(zip(best_all, mean_all), 1):
            w.writerow([idx, b, m])

    # --- wykres -----------------------------------------------------
    plt.figure()
    plt.title(label)
    plt.bar(range(1,runs+1), best_all, label="best per run")
    plt.axhline(best_all.mean(), color="red", label="mean best")
    plt.legend()
    plt.savefig(f"{label}.png", dpi=200)
    plt.close()

# ---------------- main ---------------------------------------------
if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_once(*exp)
