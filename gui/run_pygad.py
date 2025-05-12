import pygad, numpy as np, time, csv, matplotlib.pyplot as plt
from algorithm.ga_operators import (one_point_crossover, two_point_crossover,
                                    mutation_swap,
                                    arithmetic_crossover,
                                    mutation_gauss, mutation_bitflip, mutation_uniform, blend_crossover)
import algorithm.fitness_wrappers as fit
import os
from datetime import datetime


# ------- konfiguracje testowe --------------------------------------
EXPERIMENTS = [
    # Reprezentacja binarna – Hypersphere
    # ("hyper_bin_tourn_1pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "tournament", one_point_crossover, mutation_swap),
    # ("hyper_bin_tourn_1pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "tournament", one_point_crossover, mutation_bitflip),
    # ("hyper_bin_tourn_2pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "tournament", two_point_crossover, mutation_swap),
    # ("hyper_bin_tourn_2pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "tournament", two_point_crossover, mutation_bitflip),
    #
    # ("hyper_bin_rws_1pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "rws", one_point_crossover, mutation_swap),
    # ("hyper_bin_rws_1pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "rws", one_point_crossover, mutation_bitflip),
    # ("hyper_bin_rws_2pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "rws", two_point_crossover, mutation_swap),
    # ("hyper_bin_rws_2pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "rws", two_point_crossover, mutation_bitflip),
    #
    # ("hyper_bin_best_1pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "sss", one_point_crossover, mutation_swap),
    # ("hyper_bin_best_1pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "sss", one_point_crossover, mutation_bitflip),
    # ("hyper_bin_best_2pt_swap", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "sss", two_point_crossover, mutation_swap),
    # ("hyper_bin_best_2pt_bitflip", fit.fitness_hypersphere_bin, int, 200, 0, 2,
    #  "sss", two_point_crossover, mutation_bitflip),

    # Reprezentacja rzeczywista – Hypersphere
    ("hyper_real_tourn_arith_gauss", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "tournament", arithmetic_crossover, mutation_gauss),
    ("hyper_real_tourn_arith_uniform", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "tournament", arithmetic_crossover, mutation_uniform),
    ("hyper_real_tourn_blend_gauss", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "tournament", blend_crossover, mutation_gauss),
    ("hyper_real_tourn_blend_uniform", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "tournament", blend_crossover, mutation_uniform),

    # ("hyper_real_rws_arith_gauss", fit.fitness_hypersphere_real, float, 10,
    #  -5.12, 5.12, "rws", arithmetic_crossover, mutation_gauss),
    # ("hyper_real_rws_arith_uniform", fit.fitness_hypersphere_real, float, 10,
    #  -5.12, 5.12, "rws", arithmetic_crossover, mutation_uniform),
    # ("hyper_real_rws_blend_gauss", fit.fitness_hypersphere_real, float, 10,
    #  -5.12, 5.12, "rws", blend_crossover, mutation_gauss),
    # ("hyper_real_rws_blend_uniform", fit.fitness_hypersphere_real, float, 10,
    #  -5.12, 5.12, "rws", blend_crossover, mutation_uniform),

    ("hyper_real_best_arith_gauss", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "sss", arithmetic_crossover, mutation_gauss),
    ("hyper_real_best_arith_uniform", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "sss", arithmetic_crossover, mutation_uniform),
    ("hyper_real_best_blend_gauss", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "sss", blend_crossover, mutation_gauss),
    ("hyper_real_best_blend_uniform", fit.fitness_hypersphere_real, float, 10,
     -5.12, 5.12, "sss", blend_crossover, mutation_uniform),
]



def run_once(label, fitness_func, gene_type, n_genes, low, high,
             sel, xover, mut, runs=20):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{label}"
    run_dir = os.path.join("pygad_results", folder_name)
    os.makedirs(run_dir, exist_ok=True)
    prefix = os.path.join(run_dir, label)

    best_all = np.empty((runs,))
    mean_all = np.empty((runs,))

    best_run_ga = None
    best_run_fitness = float("inf")

    for i in range(runs):
        fitness_history = []

        def on_generation(ga):
            fitnesses = ga.last_generation_fitness
            fitness_history.append((np.mean(fitnesses), np.std(fitnesses)))

        gene_space = None
        if gene_type == float:
            gene_space = [{"low": low, "high": high}] * n_genes

        ga = pygad.GA(
            num_generations=100,
            sol_per_pop=50,
            num_parents_mating=20,
            num_genes=n_genes,
            fitness_func=fitness_func,
            gene_type=gene_type,
            init_range_low=low,
            init_range_high=high,
            gene_space=gene_space,
            parent_selection_type=sel,
            crossover_type=xover,
            mutation_type=mut,
            keep_elitism=1,
            K_tournament=3,
            on_generation=on_generation
        )

        t0 = time.time()
        ga.run()
        t1 = time.time()

        best = ga.best_solution()[1]
        best_all[i] = best
        mean_all[i] = np.mean(ga.best_solutions_fitness)

        # UWAGA: zapisujemy -fitness = f(x)
        print(f"{label} run {i+1}/{runs}: best={-best:.4f}  time={t1-t0:.2f}s")

        if best < best_run_fitness:
            best_run_fitness = best
            best_run_ga = ga
            best_run_history = fitness_history

    # --- CSV: zapis -fitness = f(x) ---
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "best", "mean"])
        for idx, (b, m) in enumerate(zip(best_all, mean_all), 1):
            w.writerow([idx, -b, -m])

    # --- wykres konwergencji ---
    history = np.array(best_run_ga.best_solutions_fitness)
    avg_history = np.array([m for m, _ in best_run_history])
    std_history = np.array([s for _, s in best_run_history])

    plt.figure(figsize=(8, 6))
    plt.plot(-history, label="Best Value")
    plt.plot(-avg_history, label="Mean Value")
    plt.plot(std_history, label="Std Value")
    plt.title(f"{label} – best run convergence")
    plt.xlabel("Generation")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_convergence.png", dpi=200)
    plt.close()

    # --- JSON: zapis -fitness = f(x) ---
    info = {
        "label": label,
        "gene_type": "int" if gene_type == int else "float",
        "num_genes": n_genes,
        "init_range": [low, high],
        "selection": sel,
        "crossover": xover.__name__,
        "mutation": mut.__name__,
        "runs": runs,
        "best_result": float(-best_all.min()),
        "mean_result": float(-best_all.mean()),
        "std_deviation": float(best_all.std())
    }

    with open(f"{prefix}_info.json", "w", encoding="utf-8") as f:
        import json
        json.dump(info, f, indent=2, ensure_ascii=False)




# ---------------- main ---------------------------------------------
if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_once(*exp)
