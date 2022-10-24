from kfp.v2.dsl import component


@component(base_image='python:3.10-slim')
def import_processed_features() -> str:
    import json
    import os
    with open(os.path.join('gcs', 'test_rig_features', 'processed_features.json'),
              'r') as final_features_file:
        processed_features = json.loads(final_features_file.read())
    return json.dumps(processed_features)
