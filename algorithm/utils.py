import math
import random

def calculate_chromosome_length(lower_bound, upper_bound, precision, num_variables):
    # Oblicza łączną długość chromosomu (w bitach) potrzebną do zakodowania wszystkich zmiennych
    # na podstawie dokładności i zakresu zmiennych
    bits = math.ceil(math.log2((upper_bound - lower_bound) / precision))
    return bits * num_variables

def decode(chromosome, chromosome_length, num_variables, lower_bound, upper_bound):
    # Dekoduje chromosom binarny do listy wartości rzeczywistych w zadanym zakresie
    bits_per_var = chromosome_length // num_variables
    variables = []
    for i in range(num_variables):
        start = i * bits_per_var
        end = start + bits_per_var
        bit_str = chromosome[start:end]  # wycinek odpowiadający jednej zmiennej
        int_val = int(bit_str, 2)  # zamiana binarnego ciągu na liczbę całkowitą
        max_int = 2 ** bits_per_var - 1  # maksymalna możliwa wartość binarna
        real_val = lower_bound + (upper_bound - lower_bound) * int_val / max_int
        variables.append(real_val)
    return variables

def generate_chromosome(length):
    # Generuje losowy chromosom binarny o zadanej długości
    return ''.join(random.choice('01') for _ in range(length))

def initialize_population(population_size, chromosome_length):
    # Tworzy początkową populację losowych chromosomów
    return [generate_chromosome(chromosome_length) for _ in range(population_size)]