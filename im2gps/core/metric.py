import typing as t
import numpy as np
from haversine import haversine_vector, Unit

dist_thresholds = {
    '1m': 0.001,
    '5m': 0.005,
    '10m': 0.01,
    '100m': 0.1,
    '500m': 0.5,
    '1km': 1.0,
    '5km': 5.0
}
error_thresholds = {
    '0m to 10m': (0, 0.01),
    '10m to 100m': (0.01, 0.1),
    '100m to 500m': (0.1, 0.5),
    '500m to 1km': (0.5, 1.0),
    '1km to 5km': (1.0, 5.0),
    '5km to 10km': (5.0, 10.0),
    '10km+': (10.0, float('inf'))
}


def localization_accuracy(dists: np.ndarray, thresholds: t.Dict[str, float] = None):
    if thresholds is None:
        thresholds = dist_thresholds
    accuracy = {}
    for level, dist in thresholds.items():
        accuracy[level] = np.sum(dists <= dist) / dists.size
    return accuracy


def avg_errors(dists: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = error_thresholds
    error = {}
    for level, limits in thresholds.items():
        left, right = limits
        dists_in_range = dists[np.where(np.logical_and(dists >= left, dists < right))]
        if dists_in_range.size == 0:
            error[level] = 0
        else:
            error[level] = np.average(dists_in_range)
    return error


def distribution_of_predictions_by_distance(dists: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = error_thresholds
    distribution = {}
    total = dists.shape[0]
    for level, limits in thresholds.items():
        left, right = limits
        dists_in_range = dists[np.where(np.logical_and(dists >= left, dists < right))]
        distribution[level] = dists_in_range.shape[0] / total
    return distribution


def compute_geo_distance(true_locations: np.ndarray,
                         predicted_locations: np.ndarray):
    """
    :param true_locations: list (np.ndarray) of true coordinates as (lon, lat)
    :param predicted_locations: list (np.ndarray) of predicted coordinates as (lon, lat)
    :return: np.ndarray of distances between true locations and predicted in kilometers
    """
    return haversine_vector(true_locations[:, [1, 0]], predicted_locations[:, [1, 0]], Unit.KILOMETERS)
