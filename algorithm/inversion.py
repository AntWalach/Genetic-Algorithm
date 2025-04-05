import random

def apply_inversion(chromosome, prob):
    # Operator inwersji: odwraca fragment chromosomu z prawdopodobieństwem `prob`
    if random.random() < prob:
        # Losujemy dwa różne indeksy i, j i odwracamy podciąg między nimi
        i, j = sorted(random.sample(range(len(chromosome)), 2))
        return chromosome[:i] + chromosome[i:j][::-1] + chromosome[j:]
    return chromosome  # brak zmian jeśli inwersja nie nastąpi