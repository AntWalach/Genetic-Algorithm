import os
import tkinter as tk
from tkinter import ttk, messagebox
import time
import numpy as np
from algorithm.genetic_algorithm import GeneticAlgorithm
from benchmark_functions import Hypersphere, Rana, Michalewicz
from opfunu.cec_based.cec2014 import F12014, F282014
from gui.charts import draw_charts
from gui.storage import save_results_csv, save_results_db
from tools.batch_runner import run_multiple_times

class GeneticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorytm Genetyczny - Optymalizacja")
        self.root.geometry("400x720")
        self.init_functions()
        self.create_widgets()

    def init_functions(self):
        self.hypersphere_function = Hypersphere()
        self.rana_function = Rana()
        self.hybrid_function = F12014()
        self.composition_6_function = F282014()
        self.michalewicz_function = Michalewicz()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        label_style = {"sticky": "w", "padx": 5, "pady": 3}
        entry_style = {"padx": 5, "pady": 3}

        row = 0
        self._add_combobox(main_frame, "Funkcja testowa:", row, 'test_function', ["hypersphere", "rana", "hybrid", "composition_6", "michalewicz"], label_style, callback=self._on_function_change); row += 1
        self._add_label_entry(main_frame, "Dolna granica zmiennych:", row, 'lower_bound', -5.12, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Górna granica zmiennych:", row, 'upper_bound', 5.12, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Dokładność (np. 0.01):", row, 'precision', 0.01, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Wielkość populacji:", row, 'population_size', 50, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Liczba epok:", row, 'num_epochs', 100, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Liczba zmiennych:", row, 'num_variables', 5, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Strategia elitarna (%):", row, 'elitism_rate', 10.0, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Prawdopodobieństwo krzyżowania:", row, 'crossover_prob', 0.8, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Prawdopodobieństwo mutacji:", row, 'mutation_prob', 0.05, label_style, entry_style); row += 1
        self._add_label_entry(main_frame, "Prawdopodobieństwo inwersji:", row, 'inversion_prob', 0.1, label_style, entry_style); row += 1

        self._add_combobox(main_frame, "Typ selekcji:", row, 'selection_method', ["best", "roulette", "tournament"], label_style); row += 1
        self._add_combobox(main_frame, "Typ krzyżowania:", row, 'crossover_method', ["one_point", "two_point", "uniform", "granular"], label_style); row += 1
        self._add_combobox(main_frame, "Typ mutacji:", row, 'mutation_method', ["one_point", "two_point", "boundary"], label_style); row += 1
        self._add_combobox(main_frame, "Typ optymalizacji:", row, 'optimize_type', ["minimize", "maximize"], label_style); row += 1

        ttk.Separator(main_frame, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=5); row += 1

        self.start_button = ttk.Button(main_frame, text="▶ Start", command=self.run_algorithm)
        self.start_button.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1

        self._add_batch_run_button(main_frame, row, label_style, entry_style); row += 2

        self.time_label = ttk.Label(main_frame, text="Czas wykonywania: --", anchor="center")
        self.time_label.grid(row=row, column=0, columnspan=2, pady=(10, 5))

    def _add_label_entry(self, parent, label, row, var_name, default, label_style, entry_style):
        ttk.Label(parent, text=label).grid(row=row, column=0, **label_style)
        var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
        setattr(self, var_name, var)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, **entry_style)

    def _add_combobox(self, parent, label, row, var_name, values, label_style, callback=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, **label_style)
        var = tk.StringVar(value=values[0])
        setattr(self, var_name, var)
        combobox = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combobox.grid(row=row, column=1, padx=5, pady=3)
        if callback:
            combobox.bind("<<ComboboxSelected>>", lambda e: callback())

    def _on_function_change(self):
        selected = self.test_function.get()
        if selected == "hypersphere":
            self.lower_bound.set(-5)
            self.upper_bound.set(5)
        elif selected == "rana":
            self.lower_bound.set(-512.0)
            self.upper_bound.set(512.0)
        elif selected == "michalewicz":
            self.lower_bound.set(0.0)
            self.upper_bound.set(np.pi)
        elif selected == "composition_6":
            self.lower_bound.set(-100.0)
            self.upper_bound.set(100.0)
            self.num_variables.set(30)
        elif selected == "hybrid":
            self.lower_bound.set(-100.0)
            self.upper_bound.set(100.0)
            self.num_variables.set(30)

    def run_algorithm(self):
        selected = self.test_function.get()
        minimize = self.optimize_type.get() == "minimize"

        if selected == "composition_6" and self.num_variables.get() != 30:
            messagebox.showerror("Błąd", "Funkcja Composition_6 wymaga dokładnie 30 zmiennych.")
            return

        # wybór funkcji bazowej
        if selected == "hypersphere":
            base_fn = self.hypersphere_function._evaluate
        elif selected == "rana":
            base_fn = self.rana_function._evaluate
        elif selected == "hybrid":
            base_fn = self.hybrid_function.evaluate
        elif selected == "composition_6":
            base_fn = self.composition_6_function.evaluate
        elif selected == "michalewicz":
            base_fn = self.michalewicz_function._evaluate
        else:
            messagebox.showerror("Błąd", f"Nieznana funkcja: {selected}")
            return

        def fitness_fn(x):
            value = base_fn(x)
            return value

        config = {
            "minimize": minimize,
            "precision": self.precision.get(),
            "population_size": self.population_size.get(),
            "num_epochs": self.num_epochs.get(),
            "selection_method": self.selection_method.get(),
            "crossover_method": self.crossover_method.get(),
            "crossover_prob": self.crossover_prob.get(),
            "mutation_method": self.mutation_method.get(),
            "mutation_prob": self.mutation_prob.get(),
            "inversion_prob": self.inversion_prob.get(),
            "elitism_rate": self.elitism_rate.get(),
            "lower_bound": self.lower_bound.get(),
            "upper_bound": self.upper_bound.get(),
            "num_variables": self.num_variables.get(),
            "function": selected
        }

        ga = GeneticAlgorithm(
            func=fitness_fn,
            minimize=minimize,
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

        start_time = time.time()
        solution, fitness, history, avg_history, std_history = ga.run(return_statistics=True)
        end_time = time.time()
        duration = end_time - start_time
        self.time_label.config(text=f"Czas wykonywania: {duration:.2f} s")

        save_results_csv(history, avg_history, std_history)
        save_results_db(history, avg_history, std_history)

        messagebox.showinfo("Wynik", f"Najlepsze rozwiązanie:\n{solution}\nWartość funkcji: {fitness:.5f}")
        draw_charts(history, avg_history, std_history, config=config, best_fitness=fitness, solution=solution)

    def _add_batch_run_button(self, parent, row, label_style, entry_style):
        ttk.Label(parent, text="Liczba powtórzeń:").grid(row=row, column=0, **label_style)
        self.batch_runs = tk.IntVar(value=10)
        ttk.Entry(parent, textvariable=self.batch_runs, width=6).grid(row=row, column=1, **entry_style)
        row += 1
        ttk.Button(parent, text="Uruchom wielokrotnie + eksport", command=self.run_batch).grid(row=row, column=0, columnspan=2, pady=(5, 10))

    def run_batch(self):
        selected = self.test_function.get()
        minimize = self.optimize_type.get() == "minimize"

        if selected == "composition_6" and self.num_variables.get() != 30:
            messagebox.showerror("Błąd", "Funkcja Composition_6 wymaga dokładnie 30 zmiennych.")
            return

        if selected == "hypersphere":
            base_fn = self.hypersphere_function._evaluate
        elif selected == "rana":
            base_fn = self.rana_function._evaluate
        elif selected == "hybrid":
            base_fn = self.hybrid_function.evaluate
        elif selected == "composition_6":
            base_fn = self.composition_6_function.evaluate
        elif selected == "michalewicz":
            base_fn = self.michalewicz_function._evaluate
        else:
            messagebox.showerror("Błąd", f"Nieznana funkcja: {selected}")
            return

        def fitness_fn(x):
            value = base_fn(x)
            return value

        config = {
            "minimize": minimize,
            "precision": self.precision.get(),
            "population_size": self.population_size.get(),
            "num_epochs": self.num_epochs.get(),
            "selection_method": self.selection_method.get(),
            "crossover_method": self.crossover_method.get(),
            "crossover_prob": self.crossover_prob.get(),
            "mutation_method": self.mutation_method.get(),
            "mutation_prob": self.mutation_prob.get(),
            "inversion_prob": self.inversion_prob.get(),
            "elitism_rate": self.elitism_rate.get(),
            "lower_bound": self.lower_bound.get(),
            "upper_bound": self.upper_bound.get(),
            "num_variables": self.num_variables.get()
        }

        runs = self.batch_runs.get()
        output_path = run_multiple_times(GeneticAlgorithm, config, fitness_fn, selected, num_runs=runs)
        messagebox.showinfo("Gotowe", f"Zakończono {runs} uruchomień.\nWyniki zapisane w folderze:\n{output_path}")

