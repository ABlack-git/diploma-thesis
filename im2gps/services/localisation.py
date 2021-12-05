import json
import logging
import time

import numpy as np
import typing as t
from itertools import product
from copy import copy
from dataclasses import dataclass, field, asdict

import im2gps.core.metric as metric
from im2gps.core.index import IndexConfig
from im2gps.core.localisation import LocalisationModel, LocalisationType
from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
import im2gps.utils as utils

log = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    localisation_type: LocalisationType = None
    sigma: float = None
    m: float = None
    k: int = None
    num_workers: int = 0


@dataclass
class BenchmarkParameters:
    query_dataset: DatasetEnum = None
    extended_results: bool = None
    save_result: bool = None
    save_path: str = None


@dataclass
class BenchmarkResult:
    accuracy: dict = None
    errors: dict = None
    predictions_by_dist: dict = None
    img_dist_error: dict = field(default_factory=dict)
    img_predicted_coords: dict = field(default_factory=dict)


def perform_localisation_benchmark(model_params: ModelParameters, index_config: IndexConfig,
                                   benchmark_params: BenchmarkParameters) -> BenchmarkResult:
    model = LocalisationModel(model_params.localisation_type, index_config, model_params.sigma,
                              model_params.m, model_params.k, model_params.num_workers)
    log.info(f"Localisation model: {repr(model)}")

    if index_config.index_dir is None:
        log.info("Getting training data")
        ids, coordinates, descriptors = MongoDescriptor.get_data_as_arrays(DatasetEnum.DATABASE)
        log.info("Finished getting training data")
    else:
        log.info("Getting ids and coords for training data")
        ids, coordinates = MongoDescriptor.get_ids_and_coords(DatasetEnum.DATABASE)
        descriptors = None
        log.info("Finished getting training data")

    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    log.info(f"Fitting model...")
    model.fit(ids, coordinates, descriptors)
    del ids, coordinates, descriptors
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")
    log.info(f"Model is trained.")

    log.info(f"Getting query data")
    q_ids, q_coordinates, q_descriptors = MongoDescriptor.get_data_as_arrays(dataset=benchmark_params.query_dataset)
    log.info("Finished getting query data")
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    log.info("Getting predictions...")
    predicted_locations = model.predict(q_descriptors)
    log.info("Finished getting predictions")

    if benchmark_params.extended_results:
        img_ids = q_ids
    else:
        img_ids = None

    result = _get_benchmark_results(predicted_locations, q_coordinates, img_ids)

    if benchmark_params.save_result:
        log.info("Saving test results")
        with open(benchmark_params.save_path, 'w') as f:
            json.dump(asdict(result), f)

    return result


@dataclass
class TuningParameters:
    grid: t.Dict[str, t.List[t.Any]] = None
    query_dataset: DatasetEnum = None
    save_path: str = None
    default_parameters: ModelParameters = None
    save_every: int = None
    index_configs: t.List[IndexConfig] = None


def localisation_tuning(parameters: TuningParameters):
    log.info("Getting ids and coords for training data")
    ids, coordinates = MongoDescriptor.get_ids_and_coords(DatasetEnum.DATABASE)
    log.info("Finished getting training data")
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    log.info(f"Getting query data")
    q_ids, q_coordinates, q_descriptors = MongoDescriptor.get_data_as_arrays(dataset=parameters.query_dataset)
    log.info("Finished getting query data")
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    coord_map = LocalisationModel.compute_coordinate_map(ids, coordinates)
    grid_tuples = _param_grid(parameters.grid)
    parameters_name = parameters.grid.keys()

    records = []
    for index_config in parameters.index_configs:
        for i, tup in enumerate(grid_tuples):
            start = time.time()
            experiment_parameters = _tuple_to_dict(parameters_name, tup)
            log.info(f"Tuning experiment: {i + 1}/{len(grid_tuples)}, index: {index_config.index_type.name}, "
                     f"parameters: {experiment_parameters}")
            model_params = _params_from_tuple(parameters_name, tup, parameters.default_parameters)
            model = LocalisationModel(model_params.localisation_type, index_config, model_params.sigma, model_params.m,
                                      model_params.k, model_params.num_workers)
            model.fit_from_coord_map(coord_map)
            predicted_locations = model.predict(q_descriptors)

            result = _get_benchmark_results(predicted_locations, q_coordinates)
            tuning_record = {
                "index_type": index_config.index_type.value,
                "parameters": experiment_parameters,
                "accuracy": result.accuracy,
                "errors": result.errors,
                "predictions_by_dist": result.predictions_by_dist
            }
            records.append(tuning_record)
            end = time.time()
            log.info(f"Current iteration time: {end - start:.3f}s")
            if (i + 1) % parameters.save_every == 0 or (i + 1) == len(grid_tuples):
                log.info(f"Saving tuning results. Step: {i + 1}/{len(grid_tuples)}, "
                         f"index: {index_config.index_type.name}, parameters: {experiment_parameters}")
                _save_tuning_results(records, parameters.save_path)
                log.info(f"Results saved to: {parameters.save_path}")


def _save_tuning_results(records, path):
    with open(path, "w") as f:
        json.dump(records, f)


def _param_grid(grid: dict) -> t.List[ModelParameters]:
    return [tup for tup in product(*[val for val in grid.values()])]


def _tuple_to_dict(keys, tup):
    return {k: v for k, v in zip(keys, tup)}


def _params_from_tuple(keys, tup, default_parameters):
    new_model_param = copy(default_parameters)
    for i, key in enumerate(keys):
        if key == 'localisation_type':
            val = LocalisationType(tup[i])
        else:
            val = tup[i]
        setattr(new_model_param, key, val)
    return new_model_param


def _get_benchmark_results(pred_locations, true_locations, image_ids: np.ndarray = None) -> BenchmarkResult:
    result = BenchmarkResult()
    distance_err = metric.compute_geo_distance(true_locations, pred_locations)
    result.accuracy = metric.localization_accuracy(distance_err)
    result.errors = metric.avg_errors(distance_err)
    result.predictions_by_dist = metric.distribution_of_predictions_by_distance(distance_err)
    if image_ids is not None:
        for img_id, dist_err, pred_loc in zip(image_ids, distance_err, pred_locations):
            img_id = int(img_id)
            result.img_dist_error[img_id] = dist_err
            result.img_predicted_coords[img_id] = pred_loc.tolist()
    return result
