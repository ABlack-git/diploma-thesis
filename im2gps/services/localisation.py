import json
import logging
import numpy as np
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
                              model_params.m, model_params.k)
    log.info(f"Localisation model: {repr(model)}")

    if index_config.index_dir is None:
        log.info("Getting training data")
        data = MongoDescriptor.get_as_data_dict(DatasetEnum.DATABASE)
        log.info("Finished getting training data")
    else:
        log.info("Getting ids and coords for training data")
        data = MongoDescriptor.get_ids_and_coords(DatasetEnum.DATABASE)
        log.info("Finished getting training data")

    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    log.info(f"Fitting model...")
    model.fit(data)
    del data
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")
    log.info(f"Model is trained.")

    log.info(f"Getting query data")
    query_data = MongoDescriptor.get_as_data_dict(dataset=benchmark_params.query_dataset)
    log.info("Finished getting query data")
    log.debug(f"Current memory usage: {utils.get_memory_usage():.2f}GB")

    log.info("Getting predictions...")
    predicted_locations = model.predict(query_data['descriptors'])
    log.info("Finished getting predictions")

    if benchmark_params.extended_results:
        img_ids = query_data['ids']
    else:
        img_ids = None

    result = _get_benchmark_results(predicted_locations, query_data['coordinates'], img_ids)

    if benchmark_params.save_result:
        log.info("Saving test results")
        with open(benchmark_params.save_path, 'w') as f:
            json.dump(asdict(result), f)

    return result


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
