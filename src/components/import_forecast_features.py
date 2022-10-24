from kfp.v2.dsl import component


@component(base_image='python:3.10-slim')
def import_forecast_features() -> str:
    import json
    import os
    with open(os.path.join('gcs', 'test_rig_features', 'forecast_features.json'),
              'r') as final_features_file:
        forecast_features = json.loads(final_features_file.read())
    return json.dumps(forecast_features)
