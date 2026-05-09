def nearest_neighbor(dist_matrix):

    """Implementacja algorytmu najbliższego sąsiada, stanowiąca benchmark w raporcie.
    Zwraca trasę (lista indeksów) i jej długość."""

    n = dist_matrix.shape[0]
    visited = [False] * n
    tour = [0]  # startujemy z Warszawy
    visited[0] = True

    for _ in range(n - 1):
        current = tour[-1]
        nearest = None
        nearest_dist = float('inf')

        """w pętli sprawdzamy, która odległość między ostatnio odwiedzonym miastem, 
        a wszystkimi jeszcze nie odwiedzonymi jest najmniejsza """
        
        for j in range(n):
            if not visited[j] and dist_matrix[current, j] < nearest_dist:
                nearest = j
                nearest_dist = dist_matrix[current, j]
        
        #Najbliżej położone miasto dołączamy do ścieżki i oznaczamy jako odwiedzone
        
        tour.append(nearest)
        visited[nearest] = True

    return tour