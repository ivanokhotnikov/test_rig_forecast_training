import os
from kfp.v2.dsl import component


@component(
    base_image='python:3.10-slim',
    output_component_file=os.path.join('configs', 'load_final_features.yaml'),
)
def load_final_features(data_bucket_name: str) -> str:
    import os
    import json
    with open(os.path.join('gcs', data_bucket_name, 'final_features.json'),
              'r') as final_features_file:
        final_features = json.loads(final_features_file.read())
    return json.dumps(final_features)
