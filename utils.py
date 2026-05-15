import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine(lat1, long1, lat2, long2):
    R = 6371.0

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(long2 - long1)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_distance_df(df, lat='latitude', long='longitude', name='city'):
    
    n = len(df)
    lats = df[lat].values
    longs = df[long].values

    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], longs[i], lats[j], longs[j])
            distance_matrix[i,j] = d
            distance_matrix[j,i] = d

    cities = df[name].values

    return pd.DataFrame(distance_matrix, index=cities, columns=cities)

def tour_length(tour, dist_matrix):

    """Łączna długość trasy, wliczając powrót do startu."""

    total = sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    # powrót
    total += dist_matrix[tour[-1], tour[0]]

    return total


def plot_tsp_route(df, tour, dist_matrix, title="Trasa TSP"):

    """Wizualizuje trasę TSP na mapie. Funkcja działa dla dowolnej metody zwracającej trasę jako listę indeksów miast."""

    fig, ax = plt.subplots(figsize=(8, 10))
    
    lons = df['longitude'].values
    lats = df['latitude'].values
    cities = df['city'].values
    n = len(tour)
    
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        ax.plot([lons[a], lons[b]], [lats[a], lats[b]],
                color='steelblue', alpha=0.7, linewidth=1.5, zorder=3)
    
    ax.scatter(lons, lats, s=60, color='red', alpha=0.5, zorder=5)
    
    # Miasto poczatkowe oznaczamy "gwiazdką"
    start = tour[0]
    ax.scatter(lons[start], lats[start], s=250, marker='*',
               color='gold', edgecolor='black', linewidth=1, zorder=6,
               label=f'Start: {cities[start]}')
    
    # Etykiety miast tylko dla małego zestawu
    if n <= 20:
        for i, city in enumerate(cities):
            ax.annotate(city, (lons[i], lats[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    
    length = tour_length(tour, dist_matrix)
    ax.set_title(f"{title}\nDługość trasy: {length:.1f} km")
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()