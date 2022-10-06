import os
from kfp.v2.dsl import component


@component(
    base_image='python:3.10-slim',
    output_component_file=os.path.join('configs',
                                       'import_final_features.yaml'),
)
def import_final_features() -> str:
    import os
    import json
    with open(
            os.path.join('gcs', 'test_rig_data', 'features',
                         'final_features.json'), 'r') as final_features_file:
        final_features = json.loads(final_features_file.read())
    return json.dumps(final_features)
