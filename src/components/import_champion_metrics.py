from kfp.v2.dsl import component, Output, Metrics


@component(base_image='python:3.10-slim')
def import_champion_metrics(
    feature: str,
    champion_metrics: Output[Metrics],
):
    import json
    import os

    with open(os.path.join('gcs', 'models_forecasting', f'{feature}.json'),
              'r') as registry_metrics_file:
        champion_metrics_dict = json.load(registry_metrics_file)
    with open(champion_metrics.path + '.json', 'w') as pipeline_metrics_file:
        json.dump(champion_metrics_dict, pipeline_metrics_file)
    champion_metrics.metadata['feature'] = feature
