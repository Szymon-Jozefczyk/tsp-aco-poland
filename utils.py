import pandas as pd
import numpy as np

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

    """Łączna długość trasy [km], wliczając powrót do startu."""

    total = sum(dist_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    # powrót
    total += dist_matrix[tour[-1], tour[0]]
    
    return total