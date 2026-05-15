import numpy as np

def calculate_tour_length(tour, dist_matrix):
    """
    Oblicza całkowitą długość trasy w problemie TSP.

    Parametry:
    tour : list[int]
        Lista indeksów miast, np. [0, 3, 1, 4, 2].
        Nie zawiera powrotu do miasta startowego na końcu.

    dist_matrix : np.ndarray
        Macierz odległości, gdzie dist_matrix[i, j]
        oznacza odległość między miastem i oraz miastem j.

    Zwraca:
    float
        Całkowita długość cyklu, razem z powrotem do miasta startowego.
    """
    n = len(tour)
    total_distance = 0.0

    for i in range(n):
        current_city = tour[i]
        next_city = tour[(i + 1) % n]

        total_distance += dist_matrix[current_city, next_city]

    return total_distance

def initialize_pheromones(n_cities, initial_pheromone=1.0):
    """
    Tworzy początkową macierz feromonów.

    Parametry:
    n_cities : int
        Liczba miast w problemie TSP.

    initial_pheromone : float
        Początkowa ilość feromonu na każdej krawędzi.

    Zwraca:
    np.ndarray
        Macierz feromonów o rozmiarze n_cities x n_cities.
    """
    pheromones = np.full((n_cities, n_cities), initial_pheromone, dtype=float)

    np.fill_diagonal(pheromones, 0.0)

    return pheromones

def calculate_heuristic_matrix(dist_matrix):
    """
    Tworzy macierz heurystyki dla TSP.

    Heurystyka to odwrotność odległości:
        heuristic[i, j] = 1 / dist_matrix[i, j]

    Im mniejsza odległość między miastami, tym większa atrakcyjność przejścia.

    Parametry:
    dist_matrix : np.ndarray
        Macierz odległości między miastami.

    Zwraca:
    np.ndarray
        Macierz heurystyki o takim samym rozmiarze jak dist_matrix.
    """
    heuristic = np.zeros_like(dist_matrix, dtype=float)

    mask = dist_matrix > 0
    heuristic[mask] = 1.0 / dist_matrix[mask]

    return heuristic


def choose_next_city(current_city, unvisited, pheromones, heuristic, alpha, beta, rng):
    """
    Wybiera kolejne miasto dla mrówki na podstawie reguły probabilistycznej ACO.

    Parametry:
    current_city : int
        Miasto, w którym aktualnie znajduje się mrówka.

    unvisited : set[int]
        Zbiór miast, których mrówka jeszcze nie odwiedziła.

    pheromones : np.ndarray
        Macierz feromonów.

    heuristic : np.ndarray
        Macierz heurystyki, zwykle 1 / odległość.

    alpha : float
        Waga wpływu feromonów.

    beta : float
        Waga wpływu heurystyki.

    rng : np.random.Generator
        Generator liczb losowych NumPy.

    Zwraca:
    int
        Indeks wybranego kolejnego miasta.
    """
    unvisited_array = np.array(list(unvisited))

    attractiveness = (
        pheromones[current_city, unvisited_array] ** alpha
        * heuristic[current_city, unvisited_array] ** beta
    )

    total_attractiveness = attractiveness.sum()

    if total_attractiveness == 0 or not np.isfinite(total_attractiveness):
        return rng.choice(unvisited_array)

    probabilities = attractiveness / total_attractiveness

    next_city = rng.choice(unvisited_array, p=probabilities)

    return int(next_city)

def build_ant_tour(dist_matrix, pheromones, heuristic, alpha, beta, rng, start_city=0):
    """
    Buduje pełną trasę jednej mrówki w problemie TSP.

    Mrówka zaczyna w start_city, a następnie wybiera kolejne miasta
    zgodnie z probabilistyczną regułą ACO, aż odwiedzi wszystkie miasta.

    Parametry:
    dist_matrix : np.ndarray
        Macierz odległości między miastami.

    pheromones : np.ndarray
        Macierz feromonów.

    heuristic : np.ndarray
        Macierz heurystyki.

    alpha : float
        Waga wpływu feromonów.

    beta : float
        Waga wpływu heurystyki.

    rng : np.random.Generator
        Generator liczb losowych NumPy.

    start_city : int
        Miasto startowe mrówki.

    Zwraca:
    list[int]
        Pełna trasa mrówki, bez powrotu do miasta startowego na końcu.
    """
    n_cities = dist_matrix.shape[0]

    tour = [start_city]

    unvisited = set(range(n_cities))
    unvisited.remove(start_city)

    while unvisited:
        current_city = tour[-1]

        next_city = choose_next_city(
            current_city=current_city,
            unvisited=unvisited,
            pheromones=pheromones,
            heuristic=heuristic,
            alpha=alpha,
            beta=beta,
            rng=rng
        )

        tour.append(next_city)
        unvisited.remove(next_city)

    return tour


def update_pheromones(pheromones, ant_tours, ant_lengths, rho, q):
    """
    Aktualizuje macierz feromonów po zakończeniu jednej iteracji ACO.

    Najpierw feromony parują, a potem każda mrówka zostawia feromon
    na krawędziach swojej trasy. Krótsze trasy zostawiają więcej feromonu.

    Parametry:
    pheromones : np.ndarray
        Aktualna macierz feromonów.

    ant_tours : list[list[int]]
        Lista tras zbudowanych przez mrówki w danej iteracji.

    ant_lengths : list[float]
        Lista długości tras odpowiadających trasom z ant_tours.

    rho : float
        Współczynnik parowania feromonu, zwykle z przedziału (0, 1).

    q : float
        Stała określająca ilość zostawianego feromonu.

    Zwraca:
    np.ndarray
        Zaktualizowana macierz feromonów.
    """
    pheromones *= (1.0 - rho)

    for tour, length in zip(ant_tours, ant_lengths):
        deposit = q / length
        n = len(tour)

        for i in range(n):
            city_a = tour[i]
            city_b = tour[(i + 1) % n]

            pheromones[city_a, city_b] += deposit
            pheromones[city_b, city_a] += deposit

    return pheromones

def ant_colony_optimization(
    dist_matrix,
    n_ants=30,
    n_iterations=100,
    alpha=1.0,
    beta=3.0,
    rho=0.5,
    q=100.0,
    initial_pheromone=1.0,
    seed=None,
    start_city=0
):
    """
    Główna funkcja algorytmu Ant Colony Optimization dla problemu TSP.

    Parametry:
    dist_matrix : np.ndarray
        Macierz odległości między miastami.

    n_ants : int
        Liczba mrówek w jednej iteracji.
        Każda mrówka buduje jedną pełną trasę.

    n_iterations : int
        Liczba iteracji algorytmu.

    alpha : float
        Waga wpływu feromonów na wybór kolejnego miasta.

    beta : float
        Waga wpływu heurystyki, czyli bliskości miast.

    rho : float
        Współczynnik parowania feromonu.

    q : float
        Stała określająca ilość feromonu zostawianego przez mrówki.

    initial_pheromone : float
        Początkowa wartość feromonu na każdej krawędzi.

    seed : int | None
        Ziarno generatora losowego. Przydatne do powtarzalnych eksperymentów.

    start_city : int
        Miasto startowe każdej mrówki.

    Zwraca:
    tuple
        best_tour, best_length, history

        best_tour : list[int]
            Najlepsza znaleziona trasa.

        best_length : float
            Długość najlepszej znalezionej trasy.

        history : list[float]
            Najlepsza długość trasy po każdej iteracji.
    """
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n_cities = dist_matrix.shape[0]

    rng = np.random.default_rng(seed)

    pheromones = initialize_pheromones(
        n_cities=n_cities,
        initial_pheromone=initial_pheromone
    )

    heuristic = calculate_heuristic_matrix(dist_matrix)

    best_tour = None
    best_length = float("inf")
    history = []

    for _ in range(n_iterations):
        ant_tours = []
        ant_lengths = []

        for _ in range(n_ants):
            tour = build_ant_tour(
                dist_matrix=dist_matrix,
                pheromones=pheromones,
                heuristic=heuristic,
                alpha=alpha,
                beta=beta,
                rng=rng,
                start_city=start_city
            )

            length = calculate_tour_length(tour, dist_matrix)

            ant_tours.append(tour)
            ant_lengths.append(length)

            if length < best_length:
                best_tour = tour
                best_length = length

        pheromones = update_pheromones(
            pheromones=pheromones,
            ant_tours=ant_tours,
            ant_lengths=ant_lengths,
            rho=rho,
            q=q
        )

        history.append(best_length)

    return best_tour, best_length, history



if __name__ == "__main__":
    dist_matrix = np.array([
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0]
    ])

    tour = [0, 1, 3, 2]

    print(calculate_tour_length(tour, dist_matrix))

    pheromones = initialize_pheromones(n_cities=4)
    print(pheromones)

    heuristic = calculate_heuristic_matrix(dist_matrix)
    print(heuristic)

    rng = np.random.default_rng(42)

    current_city = 0
    unvisited = {1, 2, 3}

    next_city = choose_next_city(
        current_city=current_city,
        unvisited=unvisited,
        pheromones=pheromones,
        heuristic=heuristic,
        alpha=1,
        beta=2,
        rng=rng
    )

    print("Wybrane kolejne miasto:", next_city)

    rng = np.random.default_rng(42)
    ant_tour = build_ant_tour(
        dist_matrix=dist_matrix,
        pheromones=pheromones,
        heuristic=heuristic,
        alpha=1,
        beta=2,
        rng=rng,
        start_city=0
    )

    print("Trasa mrówki:", ant_tour)
    print("Długość trasy mrówki:", calculate_tour_length(ant_tour, dist_matrix))

    ant_tours = [
        [0, 1, 3, 2],
        [0, 2, 3, 1]
    ]

    ant_lengths = [
        calculate_tour_length(ant_tours[0], dist_matrix),
        calculate_tour_length(ant_tours[1], dist_matrix)
    ]

    updated_pheromones = update_pheromones(
        pheromones=pheromones,
        ant_tours=ant_tours,
        ant_lengths=ant_lengths,
        rho=0.5,
        q=100
    )

    print("Długości tras:", ant_lengths)
    print("Feromony po aktualizacji:")
    print(updated_pheromones)

    best_tour, best_length, history = ant_colony_optimization(
        dist_matrix=dist_matrix,
        n_ants=10,
        n_iterations=20,
        alpha=1.0,
        beta=3.0,
        rho=0.5,
        q=100.0,
        seed=42,
        start_city=0
    )

    print("Najlepsza trasa ACO:", best_tour)
    print("Najlepsza długość ACO:", best_length)
    print("Historia:", history)