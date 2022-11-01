from kfp.v2.dsl import Input, Model, Metrics, component


@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=['scikit-learn', 'google-cloud-aiplatform'],
)
def upload_model_to_registry(
    feature: str,
    scaler_model: Input[Model],
    keras_model: Input[Model],
    metrics: Input[Metrics],
) -> None:
    import os
    import json
    import joblib
    from datetime import datetime

    from tensorflow import keras
    import google.cloud.aiplatform as aip
    PROJECT_ID = 'test-rig-349313'
    REGION = 'europe-west2'
    TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')
    DEPLOY_IMAGE = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-10'

    scaler = joblib.load(scaler_model.path + '.joblib')
    joblib.dump(
        scaler,
        os.path.join('gcs', 'models_forecasting', f'{feature}.joblib'),
    )
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    forecaster.save(os.path.join('gcs', 'models_forecasting', f'{feature}.h5'))
    with open(metrics.path + '.json', 'r') as pipeline_metrics_file:
        eval_metrics_dict = json.load(pipeline_metrics_file)
    with open(os.path.join('gcs', 'models_forecasting', f'{feature}.json'),
              'w') as registry_metrics_file:
        registry_metrics_file.write(json.dumps(eval_metrics_dict))
    aip.init(project=PROJECT_ID, location=REGION)
    models = [
        model.display_name for model in aip.Model.list(
            project=PROJECT_ID,
            location=REGION,
        )
    ]
    forecaster.save(
        os.path.join('gcs', 'models_forecasting', 'registry', feature))
    if feature not in models:
        model = aip.Model.upload(
            project=PROJECT_ID,
            location=REGION,
            display_name=feature,
            artifact_uri=os.path.join('gcs', 'models_forecasting', 'registry',
                                      feature),
            serving_container_image_uri=DEPLOY_IMAGE,
            is_default_version=True,
        )
    else:
        for model in aip.Model.list(project=PROJECT_ID, location=REGION):
            if model.display_name == feature:
                model = aip.Model.upload(
                    project=PROJECT_ID,
                    location=REGION,
                    parent_model=model.name,
                    display_name=feature,
                    artifact_uri=os.path.join('gcs', 'models_forecasting',
                                              'registry', feature),
                    serving_container_image_uri=DEPLOY_IMAGE,
                    is_default_version=True,
                )
                break
