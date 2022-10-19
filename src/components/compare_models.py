from kfp.v2.dsl import Input, Metrics, component


@component(base_image='python:3.10-slim')
def compare_models(
    feature: str,
    challenger_metrics: Input[Metrics],
    champion_metrics: Input[Metrics],
    evaluation_metric: str = 'root_mean_squared_error',
    absolute_difference: float = 0.0,
) -> bool:
    """
    https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/blob/main/pipelines/kfp_components/evaluation/compare_models.py
    """
    import json
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    with open(champion_metrics.path + '.json') as champ:
        champ_metrics_dict = json.load(champ)
    with open(challenger_metrics.path + '.json') as chal:
        challenger_metrics_dict = json.load(chal)
    if (evaluation_metric not in champ_metrics_dict.keys()) or (
            evaluation_metric not in challenger_metrics_dict.keys()):
        raise ValueError(f'{evaluation_metric} is not present in both metrics')
    if absolute_difference is None:
        logging.info("Since absolute_difference is None, setting it to 0.")
        absolute_difference = 0.0
    champ_val = champ_metrics_dict[evaluation_metric]
    logging.info(f'Champion metric = {champ_val:.2e}')
    chal_val = challenger_metrics_dict[evaluation_metric]
    logging.info(f'Challenger metric = {chal_val:.2e}')
    abs_diff = abs(absolute_difference)
    diff = chal_val - champ_val
    chal_is_better = (diff <= abs_diff)
    logging.info(f'Challenger is better = {chal_is_better}')
    return chal_is_better
