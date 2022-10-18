from kfp.v2.dsl import Input, Model, Metrics, component


@component(
    base_image='tensorflow/tensorflow:latest',
    packages_to_install=['joblib'],
)
def upload_model_to_registry(
    feature: str,
    scaler_model: Input[Model],
    keras_model: Input[Model],
    eval_metrics: Input[Metrics],
) -> None:
    import os
    import json
    import joblib
    from tensorflow import keras

    scaler = joblib.load(scaler_model.path + '.joblib')
    joblib.dump(
        scaler,
        os.path.join('gcs', 'models_forecasting', f'{feature}.joblib'),
    )
    forecaster = keras.models.load_model(keras_model.path + '.h5')
    forecaster.save(os.path.join('gcs', 'models_forecasting', f'{feature}.h5'))
    with open(eval_metrics.path) as pipeline_metrics_file:
        eval_metrics_dict = json.load(pipeline_metrics_file)
    with open(os.path.join('gcs', 'models_forecasting', f'{feature}.json'),
              'w') as registry_metrics_file:
        registry_metrics_file.write(json.dumps(eval_metrics_dict))
