import time
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import json
import hashlib
from datetime import datetime
import pandas as pd

def generate_run_folder_name(config, test_function):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_string = f"{test_function}_{config['selection_method']}_{config['crossover_method']}_{config['mutation_method']}_{config['num_variables']}vars_{config['num_epochs']}epochs"
    short_hash = hashlib.md5(param_string.encode()).hexdigest()[:6]
    folder_name = f"{test_function}_{config['num_variables']}vars_{config['num_epochs']}epochs_{short_hash}_{timestamp}"
    return os.path.join("batch_results", folder_name)

def save_summary_table(latest_folder, result_data):
    function_name = result_data["test_function"]
    num_vars = result_data["config"]["num_variables"]
    best_fitness = result_data["best_fitness"]

    if function_name == "hypersphere":
        optimum = 0
    elif function_name == "composition_6":
        optimum = 360
    else:
        optimum = None

    error = abs(best_fitness - optimum) if optimum is not None else None

    df = pd.DataFrame([{
        "Funkcja": function_name,
        "Liczba zmiennych": num_vars,
        "Optimum teoretyczne": optimum,
        "Najlepszy wynik": best_fitness,
        "Błąd": error
    }])

    summary_dir = os.path.join(latest_folder, "summary_table")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "tabelka_podsumowujaca.csv")
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

def run_multiple_times(ga_class, config, fitness_fn, test_function_name, num_runs=10):
    output_dir = generate_run_folder_name(config, test_function_name)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    best_fitness_overall = float("inf") if config["minimize"] else float("-inf")
    best_run_data = None
    best_run_index = -1

    for run in range(num_runs):
        run_dir = os.path.join(output_dir, f"run_{run + 1:02d}")
        os.makedirs(run_dir, exist_ok=True)

        ga = ga_class(
            func=fitness_fn,
            minimize=config["minimize"],
            precision=config["precision"],
            population_size=config["population_size"],
            num_epochs=config["num_epochs"],
            selection_method=config["selection_method"],
            crossover_method=config["crossover_method"],
            crossover_prob=config["crossover_prob"],
            mutation_method=config["mutation_method"],
            mutation_prob=config["mutation_prob"],
            inversion_prob=config["inversion_prob"],
            elitism_rate=config["elitism_rate"],
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"]
        )
        ga.num_variables = config["num_variables"]

        start = time.time()
        solution, fitness, history, avg_history, std_history = ga.run(return_statistics=True)
        end = time.time()
        duration = end - start

        with open(os.path.join(run_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Best", "Avg", "Std"])
            for epoch, (b, a, s) in enumerate(zip(history, avg_history, std_history)):
                writer.writerow([epoch + 1, b, a, s])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(history)
        ax1.set_title("Najlepsza wartość w każdej epoce")
        ax1.set_xlabel("Epoka")
        ax1.set_ylabel("Fitness")
        ax2.plot(avg_history, label="Średnia")
        ax2.plot(std_history, label="Odchylenie std.")
        ax2.set_title("Statystyki populacji")
        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("Wartość")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "charts.png"))
        plt.close(fig)

        all_results.append({
            "run": run + 1,
            "solution": solution,
            "fitness": fitness,
            "time": round(duration, 4)
        })

        if (config["minimize"] and fitness < best_fitness_overall) or \
                (not config["minimize"] and fitness > best_fitness_overall):
            best_fitness_overall = fitness
            best_run_data = (history, avg_history, std_history)
            best_run_index = run + 1

    with open(os.path.join(output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Fitness", "Time (s)", "Solution"])
        for r in all_results:
            writer.writerow([r["run"], r["fitness"], f"{r['time']:.4f}", r["solution"]])

    fitnesses = [r["fitness"] for r in all_results]
    best = min(fitnesses) if config["minimize"] else max(fitnesses)
    worst = max(fitnesses) if config["minimize"] else min(fitnesses)
    avg = float(np.mean(fitnesses))

    with open(os.path.join(output_dir, "summary_stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best fitness: {best}\n")
        f.write(f"Worst fitness: {worst}\n")
        f.write(f"Average fitness: {avg}\n")
        f.write(f"Best run: run_{best_run_index:02d}\n")

    result_data = {
        "config": config,
        "test_function": test_function_name,
        "num_runs": num_runs,
        "average_fitness": avg,
        "best_fitness": best,
        "worst_fitness": worst,
        "best_run": best_run_index,
        "results": all_results
    }

    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as jf:
        json.dump(result_data, jf, indent=2, ensure_ascii=False)

    # Zapisz tabelkę podsumowującą (punkt g)
    save_summary_table(output_dir, result_data)

    return output_dir